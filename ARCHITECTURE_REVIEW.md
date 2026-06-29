<<<<<<< HEAD
# Architecture Review — Sprint 31 Close

**Date:** 2026-06-29
**Reviewed by:** Hermes Agent (code-review + arch-review skills)
**Delta:** 20 commits, 48 files (+2704/-342)
=======
# Architectural Review — 2026-06-27

Branch reviewed: `feature/sprint-30`. Second run of `/arch-review`. Disjoint from `/code-review`
(this report contains no per-call seam-correctness or linter items by design).
>>>>>>> origin/main

---

<<<<<<< HEAD
## Axis Scorecard
=======
The architecture's core strength — clean JNI seam with single-crossing-per-render — holds. The
main problems are **governance**: the project's two most-stated strengths ("add a geometry type in
<1 day" and performance budgets) remain unguarded from the prior review, and a **shipped feature is
broken** (`type=curve` not in `VALID_TYPES`). Native governance is still aspirational: Valgrind and
compute-sanitizer return unconditional success, CUDA error checking is absent from
`menger-geometry` native code, and performance baselines are unmeasured placeholders. Biggest single
risk: three concrete bugs are visible this sprint but invisible to CI (curve CLI broken, native
handle unsafe, per-frame buffer churn) — and CI is designed not to catch any of them.
>>>>>>> origin/main

| Axis | Arc42 § | Grade | Key Gap |
|------|---------|-------|---------|
| Soundness | §5 | ✅ Guarded | Module deps enforced by ArchUnit; L-system code placed in correct packages |
| Maturity | §10/§11 | ⚠️ Partial | Tests pass but no render-determinism gate; JNI leak gate is CI-only |
| Evolvability | §9 | ⚠️ Partial | Adding a new geometry type still touches 6+ files; dispatch partially unified |
| Performance | §10.4 | ⚠️ Partial | PerfCheck exists but baselines not committed; no regression guard |

<<<<<<< HEAD
---
=======
| Axis | Health | Headline finding |
|------|--------|------------------|
| Soundness          | 🟡 yellow | Raw primitives across JNI seam; ArchUnit has module-scope holes; InteractiveEngine is a 662-line god-object |
| Maturity           | 🔴 red    | Valgrind/compute-sanitizer completely stubbed; CUDA error checks absent from menger-geometry; caustics ladder 0/8 |
| Evolvability       | 🔴 red    | `"curve"` is missing from `VALID_TYPES` — CLI `type=curve` silently broken; object-type dispatch is 6 unguarded edits |
| Performance Arch.  | 🔴 red    | PerfCheck is advisory with placeholder baselines; all GPU tunables are compile-time constants; per-frame buffer re-allocations in hot path |
>>>>>>> origin/main

## Fitness-Function Status Table

<<<<<<< HEAD
| Function | Status | Enforced? |
|----------|--------|:---:|
| Module dependency direction (ArchUnit) | ✅ Active | ✅ CI + pre-push |
| Package placement (ArchitecturePhase2Spec) | ✅ Active | ✅ CI + pre-push |
| Coverage ratchet (≥80%, max 1% drop) | ✅ Active | ✅ CI + pre-push |
| Version consistency (4 files) | ✅ Active | ✅ CI + pre-push |
| Changelog updated | ✅ Active | ✅ CI |
| JNI native leak gate (Valgrind + compute-sanitizer) | ✅ Active | ✅ CI (allow_failure) |
| CI YAML lint | ✅ Active | ✅ pre-push |
| Render determinism | ❌ Missing | ❌ None |
| Performance budget (P1/P2) | ⚠️ Advisory | ⚠️ CI (allow_failure) |
| Script parity (integration ⊇ manual) | ❌ Missing | ❌ None |
| Object-type dispatch completeness | ⚠️ Test only | ⚠️ Unit test (not blocking) |

---

## Findings

### A1 — Object-type dispatch still triplicated (Evolvability, High)

**Location:** `GeometryRegistry.scala:28-52`, `InteractiveEngine.scala:556-600`,
`RenderModeSelector.scala:18-35`
**Evidence:** The 2026-06-12 patch synced `sierpinski4d`/`hexadecachoron4d` across three
if/else chains, but the root cause (three hand-maintained dispatch tables) remains.
Adding a new type still touches 6-7 files.
**Enforcement:** Partially guarded — the completeness unit test catches missing entries, but
only after the code is written. No compile-time guard that a new `ObjectType.VALID_TYPES`
entry must have a corresponding builder.
**Fitness function:** Single source-of-truth registry table consumed by all three sites.
**See:** ARCHITECTURE_BACKLOG.md T1

### A2 — Performance is ungoverned (Performance Architecture, High)

**Location:** `scripts/benchmark.sh`, `.gitlab-ci.yml:PerfCheck`
**Evidence:** PerfCheck exists as a CI job with `allow_failure: true`. arc42 §10.4 budgets
(P1 <5s, P2 <500ms) are declared but no baselines are committed in `perf-baseline.json`.
PerfCheck runs but its results don't block anything.
**Enforcement:** Unguarded — a 10× render-time regression would not fail CI.
**Fitness function:** PerfCheck as a blocking job with committed baselines.
**See:** ARCHITECTURE_BACKLOG.md T2

### A3 — Native memory-leak gate is CI-only (Maturity, Medium)

**Location:** `.git_hooks/pre-push:343-375`, `.gitlab-ci.yml:NativeLeakCheck`
**Evidence:** The pre-push hook's Valgrind/compute-sanitizer stages are conditional on
`HAS_NATIVE` and skip gracefully when tools are not installed. In CI, `NativeLeakCheck`
runs as an advisory job (`allow_failure: true`). Native leaks introduced in a PR are
caught only when CI runs — not during local development.
**Enforcement:** Partially guarded — CI covers it but pre-push may silently skip.
**Fitness function:** Pre-push Valgrind gate should fail loudly when native files change
and tools are missing (install instructions + exit 1 instead of return 0).
**See:** ARCHITECTURE_BACKLOG.md T3

### A4 — String-based builder dispatch in LSystemSceneBuilder (Soundness, Low)

**Location:** `LSystemSceneBuilder.scala:51-55`
**Evidence:** `resolveSubBuilder` matches on `"curve"`, `"sphere"`, `"cone"` strings
with a wildcard default to `CurveSceneBuilder`. A typo in `emitRun`'s ObjectSpec
construction would silently route to the wrong builder.
**Enforcement:** Unguarded — no compile-time check that the objectType strings emitted
by the turtle match the dispatch table.
**Fitness function:** Enum or sealed trait for sub-builder type, consumed by pattern match.

### A5 — 4D presets inconsistently placed (Evolvability, Low)

**Location:** `LSystemTurtle4D.scala:253-264`, `LSystemPresets.scala`
**Evidence:** 3D presets live in `menger.objects.LSystemPresets` (shared by CLI and DSL
paths). 4D presets (`HilbertCurve4D`, `Tree4D`) are hardcoded in the `LSystemTurtle4D`
companion object, instantiated with full grammar strings — not exposed to the DSL or CLI
preset lookup. Adding a new 4D preset requires touching a different file than adding a
3D preset.
**Enforcement:** Unguarded — convention only.
**Fitness function:** Move 4D presets into `LSystemPresets` (or a 4D variant) with the
same interface.

---

## arc42 Coherence

| Section | Status | Action |
|---------|--------|--------|
| §5 Building Block View | ✅ Accurate | No changes needed — module structure unchanged |
| §9 Architectural Decisions | ⚠️ Stale | Missing: OptiX-as-sole-backend ADR (T11), L-system design decisions |
| §10 Quality Requirements | ⚠️ Stale | P1/P2 marked "Validated" but never measured; caustics ladder claims 8 rungs, 0 exist |
| §11 Risks | ✅ Accurate | JNI leak risk tracked; CI gate present |

---

## Summary

The architecture is fundamentally sound — module boundaries are enforced, test coverage
is high, and the L-system integration follows existing patterns (SceneBuilder, ObjectSpec).
Three High items (A1, A2, A3) and two Low items (A4, A5) have been added to the
architectural backlog. All are scheduled in Sprint 32 (T1, T2, T3, plus new tasks for A4/A5).
=======
| Invariant | Guarded by | Status | Action |
|-----------|-----------|--------|--------|
| version consistency (4 files) | pre-push hook | ✅ guarded | — |
| coverage ratchet ≥80% / floor 60% | pre-push + CI | ✅ guarded | — |
| scalafix / WartRemover clean | pre-push | ✅ guarded | — |
| integration suite passes | pre-push (enforcing) | ✅ guarded | — |
| no native `@native` outside `io.github.lene.optix` | ArchitectureSpec | 🟡 partial | package-scoped, not module-scoped (F2) |
| every `ObjectType.VALID_TYPES` → builder dispatches | — | 🔴 unguarded | `"curve"` missing from VALID_TYPES (F1); nominate resolver-completeness test |
| integration/manual script feature parity | — | 🔴 unguarded | 4D types missing from manual-test.sh static; caustics absent from manual |
| perf budgets P1/P2 hold | PerfCheck (allow_failure: true) | 🔴 unguarded | all baselines are unmeasured placeholder 5000.0ms (F4) |
| in-repo native (`menger-geometry`) leak-free | pre-push/CI stub | 🔴 unguarded | Valgrind + compute-sanitizer return unconditional success (F3) |
| no raw native id past the scene-builder bridge | — | 🔴 unguarded | `InstanceId` exists but is stripped before every JNI call (F2) |
| render determinism (same scene → byte-stable) | — | 🔴 unguarded | still absent |
| caustics C1–C8 / SSIM acceptance | pixel-diff only | 🔴 unguarded | 0/8 analytic rungs; SSIM never computed |
| GPU tunables configurable at runtime | — | 🔴 unguarded | all compile-time constants; zero BLOCK_SIZE/stream/batch runtime knobs (F5) |

## Findings

### 1. `"curve"` is missing from `ObjectType.VALID_TYPES` — CLI `type=curve` silently broken

**Axis:** Evolvability
**Where:** `menger-common/.../ObjectType.scala` (VALID_TYPES Set) vs `menger-app/.../dsl/SceneObject.scala:462` (Curve DSL type), `menger-app/.../cli/ObjectSpec.scala:254` (CLI parse gate)
**Impact:** Critical — a Sprint 29 feature does not work through its documented CLI interface.
The user guide documents `type=curve:control-points=…` but the CLI parser rejects `"curve"` as
"Invalid object type." The DSL path works because it bypasses the parse gate. Zero integration
tests exercise the CLI curve path.
**Effort:** minutes (add `"curve"` to `VALID_TYPES`)
**Enforcement:** unguarded
**What:** `ObjectType.VALID_TYPES` is the gate for every `ObjectSpec.parse()` call. `"curve"` was
defined in DSL `SceneObject` and wired into `GeometryRegistry.builderFor()`, `RenderModeSelector`,
and `CurveSceneBuilder` — but the set of valid CLI types was never updated. The `CurveSuite` tests
only the DSL path.
**Why it matters:** A feature shipped in Sprint 29 that users cannot invoke from the CLI. Worse:
zero tests catch it because integration tests use the `--scene` DSL path, not the CLI `type=` path.
**Suggested direction:** Add `"curve"` to `VALID_TYPES`. Add `type=curve` CLI tests to both
integration-tests.sh and manual-test.sh. Nominate a fitness function: for every type in
`VALID_TYPES`, assert at least one integration test exercises it via `type=<name>`.

### 2. JNI/GPU boundary: raw primitives, unguarded native calls, and absent CUDA error checking

**Axis:** Soundness + Maturity
**Where:** `OptiXRenderer.scala:228` (`nativeHandle: Long`), `MengerRenderer.scala:28-84` (raw `Int` instance IDs), `BaseEngine.scala:34-35` (unguarded `setDenoisingEnabled`/`setAccumulationFrames`), `menger-geometry/src/main/native/project4d.cu` (zero `cudaGetLastError`), `menger-geometry/src/main/native/caustics_ppm.cu` (1059 lines, zero CUDA checks)
**Impact:** High — the project's highest-risk boundary (JNI × CUDA × OptiX) carries the least type safety. A 3D/4D mismatch, a swapped instance ID, or an unguarded native call produces native SIGSEGV or silent geometry corruption — not a Scala-level error.
**Effort:** days
**Enforcement:** unguarded
**What:** Three sub-problems form one boundary-integrity finding: (a) `nativeHandle`, instance IDs,
and texture indices are bare `Long`/`Int` — the `InstanceId` opaque type exists within
menger-app's scene-builder layer but is stripped before every JNI call; (b) `setDenoisingEnabled`
and `setAccumulationFrames` are `public native` with no Scala `isInitialized` precondition check
(CODE_IMPROVEMENTS `M-native-no-guard`); (c) `project4d.cu` and `caustics_ppm.cu` — the
menger-geometry native code that ships with the app — have zero CUDA error checking, while the
optix-jni library's native code uses `CUDA_CHECK` macros consistently (asymmetric governance).
**Why it matters:** An async CUDA error from `project4d.cu` corrupts geometry silently. An
unguarded call to `setDenoisingEnabled` before `initialize()` causes SIGSEGV. Both bypass every
gate.
**Suggested direction:** (a) Opaque types at the JNI bridge (`opaque type GpuHandle`, phantom-typed
`InstanceId`) enforced by ArchUnit rule. (b) Scala wrapper methods with `require(isInitialized)`
for all native-accessed state mutators. (c) Add `CUDA_CHECK` macros to menger-geometry native
code, mirroring optix-jni's discipline.

### 3. Native governance is a no-op: Valgrind + compute-sanitizer stubbed, CUDA errors unchecked

**Axis:** Maturity
**Where:** `.git_hooks/pre-push:324-326, 472-474` (stubbed gate functions), `.gitlab-ci.yml` (no Valgrind/compute-sanitizer jobs), `CODE_IMPROVEMENTS.md` `L-project4d-async-error` (carried forward since Sprint 26), `ARCHITECTURE_BACKLOG.md` Task T3 (unscheduled since 2026-06-12)
**Impact:** High — arc42 §11.4 lists JNI memory leaks as "tracked and mitigated" (TR-5). The
pre-push hook prints "Valgrind: PASSED." Both claims are false: the gate returns 0 unconditionally.
The in-repo native code (`menger-geometry/src/main/native/`) has never been leak-checked.
**Effort:** Medium
**Enforcement:** unguarded (the gate is a lie)
**What:** `run_valgrind()` and the compute-sanitizer step both immediate-return with "skipped"
messages. The optix-jni standalone repo has its own checks, but menger-geometry's 12 JNIEXPORT
functions and two CUDA kernels are the highest-risk surface and have zero protection. The gap has
been known since Sprint 26 (`ARCHITECTURE_BACKLOG.md` T3, ~2 dev-days) and remains unscheduled.
**Why it matters:** The docs assert a safety net that does not exist. A native leak in
`MengerJNIBindings.cpp` would reach production undetected.
**Suggested direction:** Either (a) wire real Valgrind + compute-sanitizer checks for
menger-geometry native code, or (b) remove the stub and update arc42 §11.4 to reflect actual
coverage. Option (a) is the intended fix per ARCHITECTURE_BACKLOG T3.

### 4. Performance governance: advisory gates, unmeasured baselines, no runtime GPU tunables

**Axis:** Performance Architecture
**Where:** `.gitlab-ci.yml:451` (`PerfCheck` `allow_failure: true`), `scripts/perf-baseline.json` (all four entries = `5000.0` ms), `OptiXContext.cpp:904-924` (hardcoded stream 0, 2D launch, no block-size control)
**Impact:** High — a multiple-× regression merges green. All GPU tunables require recompilation.
arc42 §10.4 claims "✅ Baseline established" for scenes whose baseline has never been measured.
**Effort:** Medium
**Enforcement:** unguarded
**What:** The PerfCheck job landed in Sprint 28 but is advisory only. `perf-baseline.json`
contains four identical placeholder values — a sphere render at 800×600 on OptiX takes ~5-20ms,
not 5 seconds. The `benchmark.sh` `THRESHOLD=1.15` is meaningless against placeholder data. On the
GPU side, block size, stream count, batch size, and tile dimensions are all hardcoded — there is
zero runtime configurability for different GPUs or workloads.
**Why it matters:** Budgets that are never executed drift silently. arc42's "Validated ✅" marks
are fiction. Every GPU optimization requires a source-code change and recompilation of the native
library.
**Suggested direction:** (a) Measure real baselines on the CI GPU runner and commit them. (b)
Promote PerfCheck to blocking after 3 stable green runs. (c) Add environment-variable overrides
for block size and stream count (`MENGER_OPTIX_BLOCK_SIZE`, `MENGER_OPTIX_STREAMS`).

### 5. Per-frame buffer re-allocation in render hot path (IAS mode)

**Axis:** Performance Architecture
**Where:** `OptiXWrapper.cpp` render path (lines ~1400-1575): texture objects, cylinder_data, cone_data, plane_data, curve_data, menger4d_data, sierpinski4d_data, hexadecachoron4d_data
**Impact:** Medium — conditional-free `cudaFree` + `cudaMalloc` + `cudaMemcpy` on every frame for
static geometry. Comment says "if size changed" but code frees unconditionally.
**Effort:** hours
**Enforcement:** unguarded
**What:** Eight geometry-data arrays are freed and re-allocated every frame in the IAS render path.
The comment `// Reallocate GPU buffer if size changed` implies the design was to track sizes, but
the implementation unconditionally frees when `d_*_data` is non-null. For static scenes this is
unnecessary GPU allocation churn in the render hot path.
**Why it matters:** While O(1) overhead per frame and not dominant, it contradicts the comment's
design intent and wastes PCI-e bandwidth. Future higher-instance-count scenes will amplify the
waste.
**Suggested direction:** Guard re-allocation on actual size changes (track `last_*_count` and
compare before free/alloc).

### 6. InteractiveEngine is a 662-line god-object with triplicated dispatch chains

**Axis:** Soundness + Evolvability
**Where:** `menger-app/.../engines/InteractiveEngine.scala` (662 lines, 8 concerns)
**Impact:** Medium — any change to scene construction, 4D rotation fast paths, cross geometry,
screenshots, or stats risks regression across unrelated areas. The 5 4D fast-path methods are
copy-paste variants.
**Effort:** days
**Enforcement:** unguarded
**What:** Eight distinct concerns co-located with no delegation boundaries: scene construction
(lines 51-109, 457-552), 5 variant 4D rotation fast paths (lines 199-296), coordinate-cross
geometry (401-436), level validation (298-329), stats JSON (624-655), screenshot saving
(inherited), camera controller (delegated ✅), LibGDX lifecycle (required ✅). The
`buildScene4DTrackedOrFallback` dispatch chain duplicates the `GeometryRegistry.builderFor` chain
— both are if-else chains over `ObjectType` predicates with no single source of truth.
**Why it matters:** Two of the newest features (Sierpinski4D + Hexadecachoron4D) are wired in
`InteractiveEngine` but absent from `GeometryRegistry.builderFor`. A refactor to one dispatch chain
without the other silently breaks features. No ArchUnit rule or test catches this.
**Suggested direction:** Extract a `RotationFastPath` strategy object from the 5 copy-paste
methods. Single source of truth for object-type → builder dispatch (a `Map` consumed by both
InteractiveEngine and GeometryRegistry). ArchUnit rule: no class > 300 lines in the engines
package.

### 7. arc42 §9 has duplicate AD numbers; §10 quality scenarios overstate governance

**Axis:** Soundness (documentation)
**Where:** `docs/arc42/09-architectural-decisions.md` (AD-24/AD-25 each assigned to two unrelated decisions), `docs/arc42/10-quality-requirements.md` (caustics C1-C8 "placeholder" but implies validation)
**Impact:** Medium — documentation as source-of-truth is compromised when numbers conflict and
quality claims misrepresent actual gates.
**Effort:** hours
**Enforcement:** unguarded
**What:** (a) AD-24 refers to both Three-Layer Module Architecture (Sprint 25) and OptiX AI
Denoiser (Sprint 29). AD-25 refers to both libav Video Decode (Sprint 27) and Curves Primitive
(Sprint 29). Sprint 29 decisions should be AD-27 and AD-28. (b) arc42 §10 presents a detailed
C1-C8 caustics quality ladder with specific acceptance criteria — but all 8 rungs are "⬜ Not
implemented" and the only protection is an AE-diff smoke test. The phrase "to be validated" is
buried in a note while the formatting implies active governance.
**Why it matters:** Trustworthy architecture documentation is a quality goal. Duplicate AD numbers
break traceability; over-stated quality gates teach readers to disregard arc42.
**Suggested direction:** Renumber Sprint 29 ADs to 27/28. Add explicit "❌ Not Implemented"
markers to each C1-C8 rung in §10. Add a `doc-lint.sh` check for duplicate AD numbers.

### 8. DSL scene files have unrestricted JVM classpath access

**Axis:** Soundness
**Where:** `menger-app/.../dsl/` (runtime-compiled scenes share classpath with menger-app)
**Impact:** Low-Medium — a malicious or erroneous DSL scene can call `System.exit`,
`System.loadLibrary`, or access `OptiXRenderer` natives directly, bypassing all architectural
layering. CLI validation (293 lines of `CliValidation.scala`) provides strong counterpart.
**Effort:** days (for full sandboxing)
**Enforcement:** partially guarded (ArchUnit checks DSL framework packages at compile time; runtime
scenes are unrestricted)
**Why it matters:** The second untrusted-input path has no boundary enforcement. In practice, DSL
scenes are author-controlled, so the risk is low for typical usage — but undocumented.
**Suggested direction:** Document in arc42 §11 risks section. A `SecurityManager` or Java module
restriction is the proper fix but low-priority given the threat model.

## arc42 coherence

- **§5 Building Blocks** — Mostly accurate. Module graph correct; §5.2 tables list allowed
  dependencies but 3 are documented as `M-arch-*` violations with `ignore`d ArchUnit rules (AD-23).
  Layered dependency diagram is aspirational, not enforced.
- **§9 Decisions** — Duplicate AD-24/AD-25 numbers (Sprint 29 denoiser + curves need new numbers).
  Sprint 28-29 ADs are present and accurate in content.
- **§10 Quality** — Over-claims. Perf P1/P2 "Validated" are aspirational. Caustics C1-C8 ladder
  presented as active but 0/8 implemented. `M4` "<1 day to add geometry type" contradicted by the
  6-file shotgun surgery measurement (F6).
- **§11 Risks** — §11.4 JNI memory leak is tracked-and-mitigated per the text; in reality
  Valgrind + compute-sanitizer are stubbed.

## Carried forward from prior review

| Prior F# | Status | Notes |
|----------|--------|-------|
| F1 (evolvability shotgun surgery) | 🟡 Still real | 6-file surgery quantified in this review F6. `"curve"` missing from VALID_TYPES makes it worse. |
| F2 (performance ungoverned) | 🔴 Still ungoverned | PerfCheck landed sprint 28 but advisory; baselines still placeholders |
| F3 (native leak gate stubbed) | 🔴 Still stubbed | No change. T3 unscheduled since 2026-06-12. |
| F4 (raw JNI primitives) | 🔴 Still unguarded | Remains the JNI boundary's single biggest gap. F2 in this review. |
| F5 (CLAUDE.md module map) | ✅ Resolved | Fixed 2026-06-12. |
| F6 (ArchUnit package-scoped) | 🟡 Still partial | Now subsumed under F2 in this review. |
| F7 (caustics ladder 0/8) | 🔴 Still 0/8 | Governance theater continues. F7 in this review. |
| F8 (no determinism/JNI fault tests) | 🔴 Still absent | Carried forward. ARCHITECTURE_BACKLOG T7 unscheduled. |
| F9 (DSL trust boundary) | 🟡 Still partial | Documented gap, not fixed. F8 in this review. |

## Resolved since last review

- F5 (CLAUDE.md module map) — CLAUDE.md now correctly describes two in-repo modules + two external
  artifacts. arc42 §5.2.1 header updated. No doc-vs-build parity test exists yet, but the false
  claim is removed.

## Positive patterns worth preserving

1. **JNI seam is structurally sound**: single `renderWithStats` call per frame; all CUDA work stays
   in native code; `MengerRenderer` extends cleanly from `OptiXRenderer` without breaking the
   abstraction. Per-frame 4D animation stays upload-once.
2. **Material preset system**: Adding a new material type touches exactly one file
   (`Material.scala`) with ~3 lines — the project's best evolvability story.
3. **CODE_IMPROVEMENTS.md discipline**: Resolved items are deleted, not archived; findings carry
   specific file:line locations; the process actively closes tooling gaps — rare and valuable.

---

*Generated by `/arch-review` (see `.claude/commands/arch-review.md`). Prior review: 2026-06-12.*
>>>>>>> origin/main
