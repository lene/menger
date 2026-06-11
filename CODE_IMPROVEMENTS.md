# Code Quality Improvements — Open Issues

**Last Updated:** 2026-06-11

Resolved items are removed from this file entirely — git history is the record of what was fixed.

---

## High Priority

*(none)*

---

## Medium Priority

*(none — M-sceneb-validate-bypass and M-instanceid-raw-int scheduled as Task 29.7)*

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
| L4 | Comprehensive benchmarking suite | Tests | 8-10 | Partially addressed by `scripts/benchmark.sh` (sprint-28). |
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
