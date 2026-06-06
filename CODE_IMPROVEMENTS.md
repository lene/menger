# Code Quality Improvements — Open Issues

**Last Updated:** 2026-06-06

Resolved items are removed from this file entirely — git history is the record of what was fixed.

---

## High Priority

### Get CI for optix-jni and menger-common up to the same standard as menger-app

Pre-commit and pre-push hooks created in Sprint 26 (`feat/git-hooks` branches in both repos).
**Blocked on**: merging those branches to main in optix-jni and menger-common.
Once merged: parity is achieved — scalafix, ArchUnit, cppcheck (where applicable), CLAUDE.md, pre-commit, pre-push in all three repos. Remove this item then.

### Develop concept for managing three projects at once in one Agentic AI project

CLAUDE.md files authored in Sprint 26 for optix-jni and menger-common.
**Blocked on**: merging CLAUDE.md to main in optix-jni and menger-common.
Once merged: the concept is documented and in force. Remove this item then.

### H-jni-thrownew-null-class: FindClass not null-checked before ThrowNew in MengerJNIBindings.cpp

**Location**: `menger-geometry/src/main/native/MengerJNIBindings.cpp` — every catch block (7 sites)
**Impact**: High — `ThrowNew(nullptr, …)` is undefined behaviour; crashes in any environment where `java/lang/RuntimeException` is not on the class path (custom ClassLoader, stripped runtime).
**Status**: Fixed in sprint 25 via `if (exc)` guard on all sites — carry-forward note only; pattern must not recur in new JNI files.

---

## Medium Priority

### Split long running CI tasks

Pre-push hook already runs coverage and static analysis in parallel (background subshells).
Further splitting requires profiling data to identify actual bottlenecks. No evidence of a problem yet.
**Decision**: Accept as-is until profiling shows a specific bottleneck worth splitting.

### Review and revise task execution scaffold

Sub-items:
- Ensure tasks are set to in-progress and done as appropriate — this is agent discipline, not a code change; CLAUDE.md now documents the DoD requirement
- Use GitLab/GitHub for issue tracking, not in-repository .md files — Sprint .md files remain as planning artifacts; cross-repo issues tracked in GitHub/GitLab per CLAUDE.md

**Decision**: Accept current model (sprint .md for planning, GitHub/GitLab issues for bugs/features). No code change required.

---

## Low Priority

| ID | Issue | Location |
|----|-------|----------|
| L-upload-texture-file-raw-int | `uploadTextureFromFile` returns a raw negative `Int` on failure while `uploadTexture` throws `TextureUploadException`. Behavior change needed: three production callers treat negative index as "skip and continue". Deferred until fail-fast vs graceful-skip decision is made. | `optix-jni/.../OptiXTextureApi.scala:67` |
| L-menger-common-gpuproject4d-field | Tracked in [menger-common#1](https://github.com/lene/menger-common/issues/1). Remove this entry once that issue is closed. | menger-common `RenderConfig.scala` |

---

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
| H-glass-sponge-skin-diffuse | Sprint 17: `use_coverage_blend` now excludes refractive materials; `use_refractive_coverage_blend` path added; `maxRayDepth` implemented in JNI/shader. Remaining artifacts are physically correct Fresnel reflection — not a code bug. Closed. |
| L-cli-monolith: MengerCLIOptions is a 375-line monolith | Scallop registers options during construction; extracting groups into separate `self: ScallopConf =>` traits risks initialization-order issues. File is already organized with clear group separators; accept as-is. |
| L-cli-validation-density: CliValidation repetitive requires-pattern | `isSupplied` must be evaluated lazily inside `validateOpt` lambdas (after argument parsing), not eagerly in a data-driven list. The repetition is load-bearing; accept as-is. |
| M-film-maxdepth-opaque-fallback: Film opaque at max_ray_depth | Unconfirmed. `use_refractive_coverage_blend` requires `has_vertex_alpha_channel`; plain Film geometry has no vertex alpha and never enters that branch. No existing scene triggers the hypothesised fallback. |
