# Code Quality Improvements — Open Issues

**Last Updated:** 2026-06-04

Resolved items are removed from this file entirely — git history is the record of what was fixed.

---

## High Priority

### Get CI for optix-jni and menger-common up to the same standard as menger-app 

### Develop concept for managing three projects at once in one Agentic AI project

### H-jni-thrownew-null-class: FindClass not null-checked before ThrowNew in MengerJNIBindings.cpp

**Location**: `menger-geometry/src/main/native/MengerJNIBindings.cpp` — every catch block (7 sites)
**Impact**: High — `ThrowNew(nullptr, …)` is undefined behaviour; crashes in any environment where `java/lang/RuntimeException` is not on the class path (custom ClassLoader, stripped runtime).
**Effort**: 15 minutes (already fixed in sprint 25 via `if (exc)` guard on all sites)
**Status**: Fixed in sprint 25 — carried as note; pattern must not recur in new JNI files.

---

## Medium Priority

### Split long running CI tasks

### Review and revise task execution scaffold 
- ensure tasks are set to in progress and don as appropriate
- use gitlab/github for issues and task tracking, not in-repository .md files 

---

## Low Priority

| ID | Issue | Location |
|----|-------|----------|
| L-upload-texture-file-raw-int | `uploadTextureFromFile` returns a raw negative `Int` on failure while `uploadTexture` throws `TextureUploadException`. Making them consistent (throw) is a **behavior change**: the three production callers (`TextureManager`, `InteractiveEngine`, `WithAnimation`) currently treat a negative index as "skip this texture and continue". Deferred — needs a decision on fail-fast vs graceful-skip before changing the error model. | `optix-jni/.../OptiXTextureApi.scala:67` |
| L-menger-common-gpuproject4d-field | `RenderConfig.gpuProject4D` is unused after task 26.7 (GPU 4D projection is now the only path; all `menger-app` usages were removed). The field still ships in the published `menger-common_3:0.1.0` artifact. Remove it from `RenderConfig` in the menger-common repository, publish a new version, and bump the menger consumer to it. | menger-common `RenderConfig.scala` (split-out repository) |


## Tooling Gaps

*(none — all three gaps identified in the 2026-05-21 review have been closed)*

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
| H-glass-sponge-skin-diffuse | Sprint 17: `use_coverage_blend` now excludes refractive materials; `use_refractive_coverage_blend` path added (vertex_alpha × Fresnel + (1−α) × continuation); `maxRayDepth` implemented in JNI/shader. Full investigation (glass-sponge-investigation.md) found remaining visible artifacts are physically correct Fresnel reflection of the pink background at grazing angles — not a code bug. Closed. |
| L-cli-monolith: MengerCLIOptions is a 375-line monolith | Scallop registers options during construction; extracting groups into separate `self: ScallopConf =>` traits risks initialization-order issues. File is already organized with clear group separators; accept as-is. |
| L-cli-validation-density: CliValidation repetitive requires-pattern | `isSupplied` must be evaluated lazily inside `validateOpt` lambdas (after argument parsing), not eagerly in a data-driven list. The repetition is load-bearing; accept as-is. The `case Some(_)/None` branches were simplified to `case _` where safe. |
| M-film-maxdepth-opaque-fallback: Film opaque at max_ray_depth | Unconfirmed. `use_refractive_coverage_blend` requires `has_vertex_alpha_channel`; plain Film geometry (spheres, parametric) has no vertex alpha and never enters that branch. No existing scene combines Film + vertex-alpha geometry to trigger the hypothesised fallback. |
