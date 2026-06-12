# Architectural Review — 2026-06-12

Branch reviewed: `feature/sprint-27`. First run of `/arch-review`. Disjoint from `/code-review`
(this report contains no per-call seam-correctness or linter items by design).

## Summary

The architecture is structurally simple (two in-repo modules, `menger-app → menger-geometry`, over
external `io.github.lene:optix-jni` + `menger-common` artifacts) and has a genuinely good JNI seam:
upload-once 4D animation, a single native-exception helper, coherent logging. The weaknesses are not
in the code's correctness but in its **governance**: the project's two most-stated strengths —
"add a geometry type in < 1 day" (evolvability) and the performance budgets — are *unguarded and
already drifting*. Adding an object type is shotgun surgery that has silently broken in two places,
and there is no executing performance fitness function on this branch at all. Biggest single risk:
several arc42/CLAUDE.md claims (module structure, caustics ladder, leak checks, perf "Validated")
describe governance that does not exist, so the docs actively mislead.

## Axis scorecard

| Axis | Health | Headline finding |
|------|--------|------------------|
| Soundness          | 🟡 yellow | Raw primitives across the JNI seam; ArchUnit rule has a module-scope hole. |
| Maturity           | 🟡 yellow | Good gates, but the in-repo native leak check is a no-op stub and the caustics ladder is 0/8. |
| Evolvability       | 🔴 red    | "Add a geometry type" is triplicated dispatch + unenforced script parity — already drifted. |
| Performance Arch.  | 🔴 red    | No executing perf fitness function; `benchmark.sh` absent on this branch. Budgets are hope. |

## Fitness-function status

| Invariant | Guarded by | Status | Action |
|-----------|-----------|--------|--------|
| version consistency (4 files) | pre-push hook | ✅ guarded | — |
| coverage ratchet ≥80% / floor 60% | pre-push + CI | ✅ guarded | — |
| scalafix / WartRemover clean | pre-push | ✅ guarded | — |
| integration suite passes | pre-push (enforcing) | ✅ guarded | — |
| no native `@native` outside `io.github.lene.optix` | ArchitectureSpec | 🟡 partial | package-scoped, not module-scoped (F6) |
| every `ObjectType.VALID_TYPES` → exactly one builder | — | 🔴 unguarded | nominate resolver-completeness test (F1) |
| integration/manual script feature parity | — | 🔴 unguarded | nominate parity test (F1) |
| perf budgets P1/P2 hold | — | 🔴 unguarded | nominate PerfCheck job + harness (F2) |
| in-repo native (`menger-geometry`) leak-free | pre-push/CI stub | 🔴 unguarded | restore/relocate compute-sanitizer (F3) |
| no raw native id past the scene-builder bridge | — | 🔴 unguarded | nominate opaque-type + ArchUnit rule (F4) |
| render determinism (same scene → byte-stable) | — | 🔴 unguarded | nominate determinism test (F8) |
| caustics C1–C8 / SSIM acceptance | pixel-diff only | 🔴 unguarded | implement analytic rungs C1–C4 (F7) |
| 4D fast-path stays O(1) instance builds | unit test (correctness only) | 🟡 partial | nominate call-count assertion (positive #1) |

## Findings

### 1. Adding an object type is unguarded shotgun surgery — and has already drifted twice

**Axis:** Evolvability
**Where:** `menger-app/.../engines/GeometryRegistry.scala:34-59` vs `menger-app/.../engines/InteractiveEngine.scala:438-545`; `scripts/integration-tests.sh` vs `scripts/manual-test.sh`; DSL string `menger-app/.../dsl/SceneObject.scala:328`; external `menger/common/ObjectType.scala`.
**Impact:** High — the project's #3 quality goal ("Extensibility") and its `M4` fitness target ("new geometry type < 1 day, ✅ Validated") are structurally optimistic; coverage has already been lost.
**Effort:** Medium
**Enforcement:** unguarded
**What:** A new type touches ~6–7 files across two modules. Two independent dispatch chains over the same `ObjectType.is*` predicates exist (`GeometryRegistry.builderFor` and `InteractiveEngine`), and they have **diverged**: `Sierpinski4DSceneBuilder` and `Hexadecachoron4DSceneBuilder` are wired in `InteractiveEngine` (477-545) but **absent from `GeometryRegistry.builderFor`**. The same two types appear in `manual-test.sh` but are **absent from `integration-tests.sh`** — so two shipping fractal types have zero headless regression coverage, despite CLAUDE.md mandating both scripts cover every rendering feature.
**Why it matters:** Two of the newest features are silently under-dispatched and under-tested. The "add to both scripts + registry" rule is enforced only by human discipline, and it has already failed.
**Suggested direction:** (a) Single source of truth — `Map[predicate → builder factory]` consumed by both `GeometryRegistry` and `InteractiveEngine`, plus a test asserting every `ObjectType.VALID_TYPES` resolves to exactly one builder (unwired type → CI failure, not runtime). (b) A parity test grepping both scripts for `type=<...>` tokens and asserting `manual ⊇ integration` over `VALID_TYPES`.

### 2. Performance is ungoverned: no executing fitness function, harness absent on this branch

**Axis:** Performance Architecture
**Where:** `docs/arc42/10-quality-requirements.md:35-36` (P1 <5s, P2 <500ms, marked "Validated"); `.gitlab-ci.yml` (no perf job); `scripts/` (no `benchmark.sh`).
**Impact:** High — a multiple-× regression in subdivision or render time merges green.
**Effort:** Medium
**Enforcement:** unguarded
**What:** No timing job, no advisory `PerfCheck`, no baseline artifact, no `benchmark.sh` on `feature/sprint-27`. The only "performance" token in the gate is a cppcheck *static* category. Budgets defer all numbers to sprint docs and are never re-measured. *(Note: the sprint-28 worktree reportedly introduces `benchmark.sh` + a `PerfCheck` job; verify they land on the integration branch and are wired to §10.4 — until then performance governance is zero here.)*
**Why it matters:** Budgets that are never executed drift silently; "Validated ✅" rots into fiction the moment the validating sprint ends.
**Suggested direction:** Land `benchmark.sh` (host sponge-L3 gen time + an 800×600 GPU render, machine-readable + committed baseline) and an `allow_failure: true` `PerfCheck` job asserting P1/P2 against baseline; promote to blocking once stable.

### 3. In-repo native code has no memory-leak gate — the advertised one is a stub

**Axis:** Maturity
**Where:** `.git_hooks/pre-push:324-326, 472-474`; `.gitlab-ci.yml` (no valgrind/sanitizer job); claim in `docs/arc42/11-risks-and-technical-debt.md` §11.4.
**Impact:** High — the JNI/CUDA code that actually ships (`menger-geometry/src/main/native/`, `libmengergeometry.so`) is the highest-risk surface, with no leak gate in either pre-push or CI.
**Effort:** Medium
**Enforcement:** unguarded (the gate is a no-op stub)
**What:** Both Valgrind and compute-sanitizer steps print "skipped — run in the standalone repository." That standalone repo is the *external* optix-jni artifact; the Menger-specific bindings in this repo are never checked. arc42 §11.4 lists "JNI memory leaks → gradual OOM" as tracked-and-mitigated — the mitigation is stubbed out for the one module it should cover.
**Suggested direction:** A CI job running `compute-sanitizer --leak-check` over `menger-geometry` `nativeTest`, or restore the pre-push body gated on `HAS_NATIVE` changes.

### 4. GPU handles, instance ids, and dimensions cross the JNI seam as raw primitives

**Axis:** Soundness (with Maturity: `M-instanceid-raw-int`)
**Where:** `menger-geometry/.../io/github/lene/optix/MengerRenderer.scala:28-84` (`level: Int`, `instanceId: Int`, returns `Int`); `VideoLoader.scala` (`nativeHandle: AtomicLong`); scene-builder seam (`CODE_IMPROVEMENTS.md` M-instanceid-raw-int).
**Impact:** Medium-High — a 3D/4D mismatch, a swapped `instanceId`/`textureIndex`, or a stale handle is a runtime CUDA fault, not a compile error — at the project's highest-risk boundary, which has the least type safety.
**Effort:** days
**Enforcement:** unguarded
**What:** The dimension distinction that *defines* this project (3D sponge vs 4D tesseract), GPU handle identity, and instance ids all collapse to primitives across the seam. This is also the one place the otherwise-unified error model frays (`Either[String,Unit]` vs `Try[Unit]` vs raw `-1`).
**Suggested direction:** Opaque types at the Scala side of the bridge (`opaque type GpuHandle = Long`, `InstanceId`, phantom-typed `Vector[Dim]`) + an ArchUnit rule that scene-builder return types do not expose raw native ids.

### 5. CLAUDE.md's module map contradicts reality (FIXED 2026-06-12)

**Axis:** Maturity (doc coherence)
**Where:** `CLAUDE.md` opening line ("Three modules: menger-app, menger-common, optix-jni").
**Impact:** Medium — the false map misled this review's own subagents mid-flight.
**Effort:** hours (done)
**Enforcement:** unguarded (no doc-vs-build check)
**What:** `build.sbt` shows two in-repo modules (`menger-app` → `menger-geometry`) consuming `io.github.lene:optix-jni:0.1.2` and `io.github.lene:menger-common:0.1.1` as **external published Maven artifacts**. CLAUDE.md presented the two external artifacts as in-repo modules and omitted `menger-geometry` entirely.
**Correction to this review's first draft:** the original wording also blamed arc42 §5. That was an over-claim — arc42 §5 is substantially *accurate* (§5.1 diagram marks `optix-jni`/`menger-common` "published" and §5.2.9/§5.2.10 correctly distinguish published `optix-jni` from in-repo `menger-geometry`). The defect was CLAUDE.md only.
**Status:** CLAUDE.md fixed to the real two-module + external-artifact graph; arc42 §5.2.1 header clarified that `menger-common` is an external artifact. Remaining: a lightweight test asserting documented module names match `build.sbt` (backlog).

### 6. ArchUnit native-binding rule is package-scoped, not module-scoped — and `menger.geometry` is misnamed

**Axis:** Soundness
**Where:** `menger-app/src/test/scala/menger/ArchitectureSpec.scala:31-37` (path filter `.*/optix-jni/.*`), `:56-69` (native rule, package whitelist `io.github.lene.optix..`), `:173-177`; `menger-geometry/.../VideoLoader.scala` (`menger.geometry` package, 8 `@native` methods).
**Impact:** Medium — the JNI-boundary invariant ("native bindings live only where they should") is not actually enforced for in-repo code.
**Effort:** hours
**Enforcement:** partially guarded
**What:** Two rules use disagreeing scope mechanisms: a package whitelist vs a path filter that excludes `menger-geometry`'s output dir. A stray `@native` under `io.github.lene.optix` in *any* module passes; `MengerRenderer`'s 7 native overrides (in menger-geometry) are invisible to the optix-jni-scoped rule. Separately, the `menger.geometry` package — named for pure geometry — contains only a native video loader, so the "native methods outside io.github.lene.optix" rule has a latent hole.
**Suggested direction:** Make the native-binding rule module-path-based (assert bindings originate from expected module locations and only *extend* the published optix-jni surface). Rename `menger.geometry` to `menger.video` or assert it has no `@native` methods.

### 7. Caustics quality ladder C1–C8 is governance theater — 0/8 implemented, no SSIM

**Axis:** Maturity
**Where:** `docs/caustics/CAUSTICS_TEST_LADDER.md:654-661` (all 8 "⬜ Not implemented"); `docs/arc42/10-quality-requirements.md` C1–C8 + canonical scene.
**Impact:** Medium — arc42 §10 presents the ladder + canonical scene as the validation strategy for the flagship physics feature; the only real safety net is two pixel-diff integration checks (`integration-tests.sh:458, 809`), and **no SSIM is computed anywhere** despite C8's "SSIM > 0.90" gate.
**Effort:** Medium–Large
**Enforcement:** unguarded
**What:** A quality scenario you cannot fail is decoration. Energy-conservation/convergence (C5/C6) are exactly the regressions pixel-diff misses.
**Suggested direction:** Implement at least the analytic rungs C1–C4 as `AnyFlatSpec` determinism tests (no GPU needed; catches refraction-math regressions); mark the rest "not implemented" in arc42 §10 so the doc stops over-claiming.

### 8. No determinism or JNI-fault-injection test *kinds*

**Axis:** Maturity
**Where:** test dirs (no soak/fault/seed-determinism specs); `integration-tests.sh` uses ImageMagick AE-diff only.
**Impact:** Medium — the safety net is broad in unit/integration/visual KINDS but missing two that matter for a GPU/JNI system: render determinism and JNI-seam fault injection (the `M-instanceid-raw-int` `-1`→mid-frame failure has no test forcing it).
**Effort:** Medium
**Enforcement:** partially guarded (visual regression exists; the other two kinds unguarded)
**Suggested direction:** A determinism fitness function (render canonical scene twice → assert identical PNG) is cheap and locks down the reproducibility arc42 §10 claims.

### 9. DSL scene path lacks the trust-boundary validation the CLI has

**Axis:** Soundness
**Where:** `menger-app/.../cli/CliValidation.scala` (validated via Scallop) vs `menger.dsl` / `examples/dsl` (no equivalent gate); `CODE_IMPROVEMENTS.md` M-sceneb-validate-bypass.
**Impact:** Low–Medium — the second untrusted-input path reaches the native bridge with no enforced range/consistency validation.
**Effort:** hours to audit
**Enforcement:** partially guarded (CLI by convention; DSL unguarded)
**Suggested direction:** Confirm the DSL validates ranges; pin the invariant "parsed external input is validated (or wrapped in a validated type) before any `io.github.lene.optix` call."

## arc42 coherence

- **§5 Building Blocks** — accurate. Correctly marks `optix-jni`/`menger-common` as published and distinguishes in-repo `menger-geometry`. The module-map drift was in CLAUDE.md, not here (F5, fixed).
- **§9 Decisions** — mostly honored. AD-16 (OptiX-only) matches the code; AD-2 correctly marked superseded; LibGDX correctly scoped to windowing/input. The stale claim is in §10, not §9.
- **§10 Quality** — over-claims. `M4` "< 1 day ✅" contradicted by F1; caustics ladder 0/8 (F7); perf budgets "Validated" with no executing check (F2).
- **§11 Risks** — §11.4 advertises a JNI-leak mitigation that pre-push/CI stub out (F3).

## Carried forward from prior review

None — first run.

## Resolved since last review

None — first run.

## Positive patterns worth preserving

1. **JNI 4D-animation fast path is correctly upload-once** (`MengerRenderer.scala:48-84`, `InteractiveEngine.scala:222-270`, `MeshUploadPlan.scala`): per-frame animation marshals ~6 scalars per instance via `updateMenger4DProjection`, reusing recorded `instanceIds` instead of rebuilding geometry. The strongest part of the design. Guard it: assert an N-frame animation issues O(frames×instances) projection calls and O(1) instance builds.
2. **Unified native-exception handling + coherent logging**: native code routes through one `throwJavaException` helper; the Scala bridge wraps results in `Try`/`Option` with a typed `OptiXNotAvailableException`; 29 files use SLF4J vs only 4 stray `println`.
3. **Well-maintained tech-debt ledger**: `CODE_IMPROVEMENTS.md` deletes resolved items, reasons about accepted ones, and closes tooling gaps — rare discipline.

---
*Generated by `/arch-review` (see `.claude/commands/arch-review.md`).*
