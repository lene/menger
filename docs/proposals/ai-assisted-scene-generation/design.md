## Context

Menger scenes are authored in a Scala 3 DSL (`menger.dsl.*`) or via the CLI `--objects type=...`
syntax. Since v0.5.0 the renderer supports **runtime compilation of `.scala` scene files**:
`menger --scene foo.scala` feeds the file to `menger.dsl.SceneCompiler` (Dotty), which returns
`Either[String, ClassLoader]` — structured compile errors on the left, a loadable scene on the
right. `SceneLoader` then resolves the scene object via `detectObjectName` + reflection. Menger
also exports JSON render stats. **None of this changes for this feature**; it is the substrate the
AI layer stands on.

The gap is purely in authoring ergonomics: turning intent ("chrome sponge beside a tinted glass
sphere, warm three-point light, caustics") into a valid `.scala` file is a steep climb over ~20 DSL
types, implicit tuple→`Vec3` conversions, `Some()` wrappers, and per-type parameter surfaces —
followed by a slow `sbt` cycle if one authors a compiled `examples.dsl.*` object rather than a
runtime `.scala` file.

Constraints that shape this design:
- **AD-31** — OptiX is the sole rendering backend; the LLM must be an *authoring* concern, never a
  rendering path. No LLM dependency may enter `menger`, `menger-common`, or `optix-jni`.
- **arc42 §10** — quality is governed by fitness functions; any new invariant (catalog ↔ DSL parity)
  becomes an automated test.
- **Untrusted code** — runtime Dotty compilation of LLM-generated Scala is, in principle, arbitrary
  code execution on the host. This dominates the risk picture.

```
┌─ User ─────────────────────────────────────────────────────────────────────┐
│  "chrome sponge next to tinted glass sphere, warm 3-point light, caustics"  │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ natural language
                               ▼
┌─ Orchestrator (new, JVM-pure, no GPU) ──────────────────────────────────────┐
│                                                                             │
│  ┌───────────┐    ┌────────────┐    ┌───────────────────────────────────┐   │
│  │ Clarify   │───▶│ Generator  │───▶│ Static Guard (allowlist lint)     │   │
│  │ questions │    │ LLM adapter│    │ rejects imports outside allowlist │   │
│  └───────────┘    └─────┬──────┘    └─────────────┬─────────────────────┘   │
│                         │ .scala text             │ pass                    │
│                         ▼                         ▼                         │
│                 ┌───────────────────────────────────────┐                   │
│                 │ write temp scene.scala                │                   │
│                 └────────────────┬──────────────────────┘                   │
│                                  │                                          │
│            ┌─────────────────────┼─────────────────────────┐                │
│            ▼                     ▼                         ▼                │
│   ┌─────────────────┐  ┌───────────────────┐  ┌──────────────────────┐      │
│   │ compile check   │  │ preview render    │  │ collect feedback     │      │
│   │ --validate-scene│  │ low-res / denoised│  │ stats JSON + PNG +   │      │
│   │ (no GPU)        │  │ subprocess (GPU)  │  │ optional vision      │      │
│   └────────┬────────┘  └─────────┬─────────┘  └──────────┬───────────┘      │
│            │ error? ≤3 retry     │                       │ user text/vision │
│            ▼                     ▼                       ▼                  │
│       ◀──────────── Refine loop (Feedback sealed trait) ─────────────▶      │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ accepted scene.scala
                                  ▼
                       ┌────────────────────────┐
                       │ menger --scene         │  (existing, unchanged)
                       │ scene.scala   (GPU)    │
                       └────────────────────────┘
```

## Goals / Non-Goals

**Goals:**
- Turn a natural-language description into a `.scala` DSL file that compiles and renders via the
  existing `--scene <file>.scala` path.
- Ask targeted clarifying questions when intent is ambiguous, instead of silently guessing.
- Auto-correct compile errors in a bounded closed loop, so the user never hand-fixes an LLM typo.
- Refine the scene from structured feedback (compile errors, render stats) and optional human/vision
  input until the result matches expectations.
- Keep the LLM entirely outside the renderer and its published artifacts.
- Prevent arbitrary code execution from generated scenes via a static guard + sandboxed execution.

**Non-Goals:**
- No new rendering capability, geometry type, material, or shader. The DSL surface is unchanged.
- No LLM inside `menger`, `menger-common`, or `optix-jni`.
- No pixel-exact automated scene-quality judgment — acceptance is ultimately user-gated.
- No replacement for hand-authoring by experts; this is an on-ramp, not a constraint system.

## Decisions

### D1 — Orchestrator is a new JVM-pure module in a new repo, not inside the renderer

The AI orchestration lives in a new git repository (working name `menger-ai`) that depends on
**none** of `optix-jni` / GPU / native code. It talks to menger via two narrow contracts only: the
**CLI subprocess** (`menger --scene x.scala`) and the **tool-catalog** (a data description of the
DSL). This keeps the module CI-green without a GPU and respects AD-31.

**Alternatives considered:**
- *Inside `menger-app`*: rejected — couples an LLM/network dependency to the renderer build and
  violates the "LLM is authoring, not rendering" boundary.
- *Standalone non-JVM tool (Python/Node)*: loses shared Scala types and the ability to run an
  in-process catalog-parity test against `menger.dsl.*` via reflection. Reconsider only if LLM SDK
  needs (vision, tool-calling) make a JVM HTTP client impractical — see D2.
- *Pure agent-workflow / prompt artifact (no compiled code)*: too weak for the safety guard (D4),
  which must be deterministic compiled logic, not prompt instruction.

### D2 — Provider-agnostic LLM adapter over plain HTTP

Define a `SceneGenerator` interface (`clarify`, `generate`, `refine`) with adapter implementations.
The initial adapter calls an LLM via **plain HTTPS** (no heavyweight SDK dependency), so the module
stays JVM-native and the provider is swappable. A scripted **stub adapter** drives the loop in unit
tests with no network.

**Alternatives:** vendor SDK (richer features but adds a dependency and pins a provider); in-process
local model (no network, but GPU/CPU cost and ops burden — viable later, behind the same interface).

### D3 — Tool-catalog: hand-authored with NL hints, enforced by a parity fitness function

The generator consults a **catalog** describing every DSL object type, material preset, light,
camera option, and feature block (caustics/fog/env-map/tone-mapping), each with parameter
name/type/default/range **plus natural-language hints** that reflection cannot provide. The catalog
is hand-authored (YAML) because the hints are what make generation good.

Drift is the enemy: a catalog that promises a type the DSL removed (or omits one it added) silently
degrades generation. A **parity fitness function** — same pattern as the existing `ScriptParitySuite`
— asserts the catalog's type/preset set equals the set discoverable by reflection over
`menger.dsl.*`. Divergence fails CI.

**Alternatives:** auto-generate the catalog from reflection (loses NL hints); maintain catalog
alongside the DSL reference doc and parity-check against the doc (fragile — doc drifts too).

### D4 — Static guard + sandboxed execution (the dominant risk control)

Runtime-compiling LLM-generated Scala is arbitrary code execution. Two layers:

1. **Static guard (deterministic, compiled).** Before any file is compiled, the orchestrator lints
   the generated source against an **allowlist**: permitted `import` roots are exactly
   `menger.dsl.*`, `scala.language.*`, `scala.math`, `examples.dsl.common.*` (the reusable
   materials/lighting library). Any other import, or references to `Runtime` / `sys.process` /
   `java.io` / `java.net` / `System.exit` / reflection, **rejects the file and feeds the rejection
   back to the generator** as a compile-style error. The guard is a parser-level check, not a
   prompt instruction — it cannot be talked out of it.
2. **Sandboxed execution.** The compile/render subprocess runs in the project's existing container
   footprint (CI already runs under Docker for GPU). Documentation recommends never running the loop
   against untrusted input outside a container.

**Alternatives:** a custom Scala compiler plugin/SecurityManager (heavy, fragile, SecurityManager is
deprecated); relying on prompt instructions alone (not a security boundary).

### D5 — New `--validate-scene` CLI flag for GPU-free validity checks

Add a menger CLI flag `--validate-scene <file.scala>` that runs `SceneCompiler` + `SceneLoader`
(load only) and exits non-zero with the `Left` message on failure, **without rendering and without a
GPU**. This lets the refine loop's compile-check step run fast, in CI, and on machines without a
GPU. It is a small, generally-useful addition independent of this feature.

**Alternatives:** call `SceneCompiler` in-process from `menger-ai` (faster, but couples the module's
classpath to `menger-app` and to Dotty at runtime — undermines the "no renderer coupling" boundary).
Decide between subprocess `--validate-scene` and in-process call in tasks; the CLI flag is
recommended because it works regardless of module boundaries.

### D6 — Feedback as a sealed trait; vision is opt-in

The refine loop consumes a sealed `Feedback` type:
- `CompileError(msg)` — from `--validate-scene` / `SceneCompiler` `Left`.
- `RenderStats(json, image)` — menger's JSON render stats plus the rendered output PNG.
- `UserText(text)` — human-in-the-loop ("too dark", "make the glass bluer"). Always available.
- `VisionCritique(image, analysis)` — multimodal LLM inspects the rendered PNG. **Opt-in** behind a
  vision-capable adapter; the text path works without it.

**Alternatives:** require vision for refinement (raises the bar — needs a multimodal provider and
costs more per iteration); pure metrics-driven autonomous refinement (rejected — Non-Goal: no
pixel-exact automated scene-quality judgment).

### D7 — Preview-then-final rendering

Iteration renders use fast preview settings (low resolution, fewer accumulation frames, denoised) so
the loop is responsive; a final high-quality render runs once the user accepts. Render settings are
already first-class in menger, so the orchestrator simply chooses preview vs final presets.

## Risks / Trade-offs

- **[Arbitrary code execution from generated Scala]** → D4 static allowlist guard (deterministic,
  parser-level) + containerized subprocess. Residual risk: a cleverly-formed allowed construct that
  still does harm; mitigated by container. Documented as the primary safety note.
- **[LLM hallucinates non-existent DSL types/params]** → catalog constrains generation (D3) and the
  compile loop (D5) catches the rest within ≤3 auto-fix rounds before escalating to the user.
- **[Catalog drifts from the real DSL]** → parity fitness function (D3), mirroring `ScriptParitySuite`.
- **[LLM cost / latency in the loop]** → bounded auto-fix retries (≤3), bounded refinement rounds,
  preview-render budget (D7), optional response caching. Fail open to "ask the user" on budget
  exhaustion.
- **[Non-deterministic path-traced renders confuse refinement]** → previews use a fixed seed and low
  sample counts; acceptance is user-gated, never pixel-exact (non-goal).
- **[Multimodal vision unavailable or costly]** → vision critique is opt-in (D6); the text-feedback
  and compile-error paths are fully functional without it.
- **[New module raises the build/CI surface]** → `menger-ai` is JVM-pure (no GPU/native), so it adds
  a cheap CI lane; the parity test is the only cross-module coupling.

## Migration Plan

Additive — no existing behavior changes except the new optional `--validate-scene` flag (D5), which
is purely additive and independently useful. The `menger-ai` module and catalog are new; the catalog
parity test starts green because the catalog is authored against the current DSL. Rollback is simply
removing the module and flag.

## Open Questions

1. **Initial LLM provider?** Anthropic, DeepSeek or a local model — D2 keeps this swappable; the
   first adapter is picked at task time.
2. **Teaching the LLM the DSL surface** — via prompt, few-shot examples, or RAG capabilities? D2 keeps this
   swappable; the first adapter is picked at task time.
3. **Vision critique in v1 or deferred?** Affects scope (one adapter + image plumbing) and the
   "matches expectations" definition. Recommend deferring to a follow-up; ship text + compile-error
   refinement first.
4. **Additionally call `SceneCompiler` in-process for speed?** The `--validate-scene` CLI path is
   now spec-mandated (see "GPU-free scene validation"). The open part is whether `menger-ai` should
   *also* call `SceneCompiler` in-process to skip subprocess overhead — at the cost of coupling its
   classpath to Dotty / `menger-app` (see D5). Default: subprocess only.
5. **Catalog location** — inside `menger-ai`, or co-located with the DSL reference so it ships with
   menger and is parity-checked there?
6. **Does refinement also tune render settings** (resolution, samples, denoise) or only scene
   contents (objects, materials, lights)? Recommend: render settings are an orchestrator-level
   concern (preview vs final, D7), not something the LLM mutates per user feedback.
7. **Should the generated scene be a runtime `.scala` file only, or also offer to drop a compiled
   `examples.dsl.*` object** for users who want type-checking and version control? Recommend runtime
   `.scala` first (fast loop); compiled-object export as a later convenience.
8. **Manual user edits to generated DSL** — what is the workflow to enable them, and how does the 
   orchestrator handle them? Recommend: user edits are allowed, but the orchestrator treats them as
   a new input and re-runs the compile-check + render-refine loop from scratch.