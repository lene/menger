# Code Quality Improvements — Open Issues

**Last Updated:** 2026-06-07 (Sprint 26 review)

Resolved items are removed from this file entirely — git history is the record of what was fixed.

---

## High Priority

### H-jni-pending-exception-ub: GetFieldID failure leaves pending JNI exception; returning -1 with it is UB

**Location**: `menger-geometry/src/main/native/MengerJNIBindings.cpp:39–44` (`getWrapper`)
**Impact**: High — per JNI spec, calling further JNI functions or returning to Java with a pending exception is undefined behaviour. `GetFieldID` on a missing field sets a `NoSuchFieldError` pending and returns `nullptr`; `getWrapper` checks for `nullptr` and returns `nullptr` to the caller, which returns `-1` to Java. The pending exception is never cleared (`ExceptionClear`) or propagated explicitly. Some JVMs will crash on re-entry.
**Fix**: After `GetFieldID` returns `nullptr`, call `env->ExceptionDescribe(); env->ExceptionClear();` before returning `nullptr` from `getWrapper`. Or rethrow as a C++ exception that the outer `catch` block will re-raise through `ThrowNew`.

---

### H-jni-release-array-mode: ReleaseFloatArrayElements uses mode `0` on exception path

**Location**: `menger-geometry/src/main/native/MengerJNIBindings.cpp:79`
**Impact**: High — mode `0` copies native memory back to the Java array before freeing. On the exception path inside `addRecursiveIASSpongeInstanceNative`, the buffer may be partially written by the interrupted operation. The correct mode on error paths is `JNI_ABORT` (2), which discards without write-back and avoids corrupting the caller's array.
**Fix**: Change `env->ReleaseFloatArrayElements(transform, transformArr, 0)` in the inner `catch (...)` rethrow block (line 79) to `env->ReleaseFloatArrayElements(transform, transformArr, JNI_ABORT)`.

---

### H-caustics-raygen-leak: temp_raygen_record leaks on exception in launchCausticsPass

**Location**: `menger-geometry/src/main/native/CausticsRenderer.cpp:34–51`
**Impact**: High — `temp_raygen_record` is allocated at line 34 via `createTempRaygenSBTRecord`. `optix_context.launch()` at line 42 can throw (it wraps OptiX calls via `CUDA_CHECK` which throws `std::runtime_error`). If it does, execution jumps past `freeTempRaygenSBTRecord` at line 51, permanently leaking the GPU allocation. Each render frame with caustics allocates this record; repeated throws would exhaust GPU memory.
**Fix**: Wrap `temp_raygen_record` in a RAII guard (e.g., a scoped `auto _ = std::unique_ptr` with a custom deleter calling `freeTempRaygenSBTRecord`) or add a `try/catch` that frees it on exception before rethrowing.

---

## Medium Priority

### M-sceneb-validate-bypass: buildSceneFromConfigs skips validate(); SceneBuilder has split error types

**Location**: `menger-app/src/main/scala/menger/engines/BaseEngine.scala:104–106`, `menger-app/src/main/scala/menger/engines/scene/SceneBuilder.scala:47,62`
**Impact**: Medium — two related issues in the same seam:
1. `buildSceneFromConfigs` (BaseEngine.scala:104) calls `builder.buildScene(...)` directly without calling `validate()` first. `buildSceneFromSpecs` at line 88 does call `validate()`. The trait's contract says validate must succeed before buildScene; one caller respects this, the other doesn't. Unvalidated input reaches GPU instance allocation.
2. `SceneBuilder.validate` returns `Either[String, Unit]` (string error, no stack trace) while `buildScene` returns `Try[Unit]`. Callers in BaseEngine must bridge these incompatible types; the string error from `validate` is swallowed into a `RuntimeException` message without structured context.
**Fix**: Call `validate()` inside `buildSceneFromConfigs` before `buildScene`, or make `buildScene` call `validate` internally as a precondition. Unify the error type (`Try` or `Either` — pick one).

---

### M-instanceid-raw-int: Inconsistent -1 handling for native instance IDs across scene builders

**Location**: `menger-app/src/main/scala/menger/engines/scene/` (CubeSpongeSceneBuilder.scala, ConeSceneBuilder.scala, Menger4DSceneBuilder.scala and others), `io.github.lene.optix.OptiXMeshApi` (in optix-jni)
**Impact**: Medium — `add*Instance` methods return raw `Int` where -1 means failure. Each builder handles this differently: some raise on -1, some silently skip, some check but don't propagate. If `add*4DInstance` returns -1 at scene-build time and the value flows to `update*4DProjection` in the render loop, a `require` fires mid-frame (`IllegalArgumentException`).
**Fix**: Wrap native instance IDs in an opaque type (`opaque type InstanceId = Int`) and return `Option[InstanceId]` or `Try[InstanceId]` from the Scala API layer. Enforce the -1→failure translation once at the boundary rather than in every caller.

---

## Low Priority

| ID | Issue | Location |
|----|-------|----------|
| L-upload-texture-file-raw-int | `uploadTextureFromFile` returns a raw negative `Int` on failure while `uploadTexture` throws `TextureUploadException`. Behavior change needed: three production callers treat negative index as "skip and continue". Deferred until fail-fast vs graceful-skip decision is made. | `optix-jni/.../OptiXTextureApi.scala:67` |
| L-project4d-async-error | `cudaGetLastError()` at `project4d.cu:147` only captures launch-configuration errors (invalid grid/block dims), not async kernel execution errors. The required `cudaDeviceSynchronize` is in the caller, not the callee — a future caller that omits the sync would silently miss kernel errors. Document the contract or move the sync inside `launchProject4DQuadsKernel`. | `menger-geometry/src/main/native/project4d.cu:147` |

---

## Tooling Gaps

*(none — all gaps identified in the 2026-05-21 review have been closed)*

| Tool | Status | Where |
|------|--------|--------|
| ArchUnit | Closed — Phase 1: 14 active rules in `ArchitectureSpec.scala`; Phase 2: 5 active + 4 ignored-with-blockers in `ArchitecturePhase2Spec.scala`; wired into `sbt test` | `menger-app/src/test/scala/menger/ArchitectureSpec.scala`, `ArchitecturePhase2Spec.scala` |
| cppcheck | Closed — runs in pre-push hook + CI `Test:Cppcheck` job | `.cppcheck-suppress`, `.git_hooks/pre-push`, `.gitlab-ci.yml` |
| clang-tidy | Closed — `compile_commands.json` enabled via CMake; runs in pre-push hook + CI `Test:ClangTidy` job | `.clang-tidy`, `CMakeLists.txt`, `.git_hooks/pre-push`, `.gitlab-ci.yml` |

---

## Feature Ideas (Sprint 20+)

These are deferred feature ideas, not defects.

| ID | Idea | Location | Est. Hours |
|----|------|----------|------------|
| L2 | Metrics and Telemetry | New feature | 6-8 |
| L3 | Scene graph abstraction | Architecture | 10-12 |
| L4 | Comprehensive benchmarking suite | Tests | 8-10 |
| L5 | Plugin system for geometry types | Architecture | 12-15 |

---

## Accepted / Deferred

Issues that were investigated and consciously accepted:

| Item | Decision |
|------|----------|
| Mutable state in LibGDX integration | Required by LibGDX framework |
| M11: Input controller mutable state | Well-structured; encapsulation adds complexity without benefit |
| L11: Exceptions in CudaBuffer (CudaBuffer.h:77,89) | Correct pattern at JNI boundaries |
| OptiX cache management | Works correctly |
| Caustics algorithm limitations | Resolved in Sprint 14 (PPM implemented; remaining limits documented in `docs/guide/advanced.md` §Caustics) |
| L-film-blend: blendFresnelColorsRGBAndSetPayload duplicates scalar body | GPU perf trade-off; acceptable if documented |
| OptiX DSL runtime evaluation | Deferred (Sprint 15) |
| H-glass-sponge-skin-diffuse | Sprint 17: `use_coverage_blend` now excludes refractive materials; `use_refractive_coverage_blend` path added; `maxRayDepth` implemented in JNI/shader. Remaining artifacts are physically correct Fresnel reflection — not a code bug. Closed. |
| L-cli-monolith: MengerCLIOptions is a 375-line monolith | Scallop registers options during construction; extracting groups into separate `self: ScallopConf =>` traits risks initialization-order issues. File is already organized with clear group separators; accept as-is. |
| L-cli-validation-density: CliValidation repetitive requires-pattern | `isSupplied` must be evaluated lazily inside `validateOpt` lambdas (after argument parsing), not eagerly in a data-driven list. The repetition is load-bearing; accept as-is. |
| M-film-maxdepth-opaque-fallback: Film opaque at max_ray_depth | Unconfirmed. `use_refractive_coverage_blend` requires `has_vertex_alpha_channel`; plain Film geometry has no vertex alpha and never enters that branch. No existing scene triggers the hypothesised fallback. |
| H-jni-thrownew-null-class | Fixed Sprint 25 — `if (exc)` guards added at all 7 ThrowNew sites in MengerJNIBindings.cpp. Pattern must not recur in new JNI files. |
| H-scene4d-race | `scene4DCache` get+set non-atomic, but all callers (render + key handlers) are on the LibGDX GL thread. `AtomicReference` is for cross-thread visibility only. Invariant documented at declaration site (`InteractiveEngine.scala:111`). |
| Split long running CI tasks | Pre-push hook already runs coverage+static analysis in parallel. No profiling data shows further bottleneck. Accept as-is. |
| Review and revise task execution scaffold | Sprint .md for planning; GitHub/GitLab issues for bugs/features. CLAUDE.md documents DoD. No code change required. |
