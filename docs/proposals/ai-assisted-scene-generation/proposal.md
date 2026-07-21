## Why

Authoring a Menger scene today requires writing valid Scala DSL against a large, evolving type
surface (objects, materials, lights, 4D projections, caustics, fog, env maps, tone mapping) — or
falling back to the verbose CLI `--objects type=...` syntax. The learning cliff is steep: a user
with a clear mental picture ("a chrome sponge next to a tinted glass sphere on a checkered floor
with warm three-point lighting and caustics") must still map that intent onto ~20 DSL types, get
tuple-vs-`Vec3` conversions right, remember `Some()` wrappers, and recompile.

The enabling infrastructure already exists and is unused for this purpose: `menger --scene
foo.scala` performs **runtime Dotty compilation** of a `.scala` DSL file
(`menger.dsl.SceneCompiler` / `SceneLoader`) and returns structured compile errors as
`Either[String, ...]`. That means a tight author→compile→render→refine loop is achievable *without*
a full `sbt` cycle per iteration and *without* pulling LLM concerns into the renderer. This change
adds the missing piece: an AI orchestration layer that turns a natural-language description into a
valid DSL file, asks clarifying questions when intent is ambiguous, and refines the scene against
feedback until it matches expectations.

## What Changes

- **Natural-language → DSL generation.** A user describes a scene in prose; the system produces a
  syntactically-valid `.scala` DSL file targeting the existing `menger.dsl.*` surface, consumable
  directly by `menger --scene <file>.scala`.
- **Clarification protocol.** When the description is ambiguous or underspecified (missing scale,
  unspecified material, contradictory lighting), the system asks targeted clarifying questions
  before generating — rather than guessing silently.
- **Compile-error feedback loop.** Generated DSL is validated through the existing runtime
  `SceneCompiler`. Compile errors (returned as `Left(msg)`) are fed back to the generator and
  auto-corrected in a closed loop, so the user never sees a hand-written compile failure.
- **Render-based refinement.** Once a scene compiles and renders, the system produces structured
  feedback (render success/failure, JSON stats already exported by menger, and the rendered image —
  captured always, inspected only if a vision adapter is configured) and refines scene parameters in
  response to user feedback ("too dark", "sponge too big", "make the glass bluer").
- **Tool-catalog / DSL schema as a managed artifact.** The generator consults a machine-readable
  description of the DSL (object types, material presets, parameter names/ranges/defaults) kept in
  sync with the renderer. Parity between this catalog and the real DSL is enforced by a fitness
  function (same pattern as the existing `ScriptParitySuite`).
- **Out of scope (explicitly):** no LLM dependency enters the `menger` renderer or its published
  artifacts (`menger-common`, `optix-jni`). The AI layer is an orchestration tool that *calls*
  menger as a subprocess / library, preserving the OptiX-as-sole-backend boundary (AD-31).

## Capabilities

### New Capabilities

- `ai-scene-generation`: Turning a natural-language scene description into a valid `.scala` DSL
  file, including the clarification protocol for ambiguous/underspecified intent and the
  tool-catalog that constrains generation to the real DSL surface.
- `scene-refinement-loop`: The closed feedback loop that validates generated DSL via the existing
  runtime compiler, renders it, gathers structured feedback (compile errors, render stats, optional
  image), and drives parameter refinement until the result matches expectations.

### Modified Capabilities

*(none — `openspec/specs/` is currently empty; this is the first spec-bearing change. The runtime
scene-compilation and JSON-stats export it relies on already exist and require no requirement-level
changes.)*

## Impact

- **New code:** An AI orchestration component (module / standalone tool / agent workflow — location
  decided in `design.md`). No changes to the renderer's ray-tracing or JNI paths.
- **Dependencies (new):** An LLM is required — either an external API (network + API key) or a local
  model. Multimodal (vision) capability is needed *only* if the refinement loop is to inspect
  rendered images autonomously; a human-in-the-loop text-feedback path needs no vision. This is the
  central design decision and is left to `design.md`.
- **Existing assets leveraged (no change):** `menger.dsl.SceneCompiler` (runtime Dotty compile),
  `SceneLoader` (`.scala` file → `LoadedScene`), `--scene <file>.scala` CLI path, JSON render-stats
  export.
- **New fitness function:** tool-catalog ↔ real-DSL parity check, mirroring `ScriptParitySuite`.
- **Architecture boundary:** AD-31 (OptiX as sole rendering backend) is respected — the LLM is an
  *authoring* tool, not a rendering backend. If the orchestration is in-tree, arc42 §9 should record
  the decision that the LLM is an external, swappable authoring concern.
- **Docs:** new user-facing doc for the AI authoring workflow; the DSL reference remains the source
  of truth the catalog is derived from.
