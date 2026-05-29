# Sprint 24: optix-jni Architecture, Design & Foundation

**Sprint:** 24 - optix-jni Decoupling: Architecture & Foundation
**Status:** Not Started
**Estimate:** ~20 hours
**Branch:** `feature/sprint-24`
**Dependencies:** None (refactoring only, no new features)

---

## Goal

Establish the architectural foundation for decoupling `optix-jni` into a generic,
publishable JVM/OptiX library (`io.github.lene.optix`) and a Menger-specific geometry
layer (`menger-geometry`). Deliver the package rename, `BaseParams`/`MengerParams`
struct split, module skeleton, and publication pipeline — ready for the implementation
sprint (Sprint 25).

---

## Target Architecture

```
[io.github.lene:optix-jni]       Generic OptiX/CUDA renderer + full OptiX API wrapper
                                   - Generic geometry: sphere, cylinder, cone, plane, mesh
                                   - Generic shaders: raygen, miss, hit_*, optix_shaders
                                   - BaseParams struct (no Menger-specific fields)
                                   - Publication: GitLab Package Registry + Maven Central
                                   - Depends on: io.github.lene:menger-common

[io.github.lene:menger-common]    Rendering primitives: Color, Vector, ImageSize, Light, etc.
                                   - Publication: GitLab Package Registry + Maven Central

[menger-geometry]                  In-repo module (not published)
                                   - 4D fractal geometry (Menger, Sierpinski, Hexadecachoron)
                                   - Recursive IAS sponge
                                   - CausticsRenderer
                                   - MengerParams struct (extends BaseParams)
                                   - Depends on: optix-jni (published artifact)

[menger-app]                       Depends on: menger-geometry, menger-common
```

The `BaseParams`/`MengerParams` split uses the standard C struct extension pattern:
`MengerParams` has `BaseParams base` as its first member. Base shaders cast the launch
params pointer to `BaseParams*`; 4D shaders cast to `MengerParams*`. The JNI launch
call in menger-geometry passes `sizeof(MengerParams)` to `optixLaunch`. This is
layout-safe because `BaseParams` is at offset 0 in `MengerParams`.

---

## Success Criteria

- [ ] Architecture review complete and design doc written (see Task 24.1)
- [ ] Package renamed to `io.github.lene.optix` — compiles and all tests pass
- [ ] `Params` renamed to `BaseParams` — all base shaders updated, tests pass
- [ ] `menger-geometry` sbt subproject scaffolded (compiles, empty)
- [ ] `MengerParams` struct defined and CUDA compilation validated
- [ ] `menger-common` publishable to GitLab Package Registry + Maven Central (CI job green)
- [ ] `optix-jni` publishable to GitLab Package Registry + Maven Central (CI job green)
- [ ] All existing tests pass

---

## Tasks

### Task 24.1: Architecture Review & Design Document

**Estimate:** 4h

Before touching code, produce a written design document covering:

1. **3-layer architecture** — formal definition of what belongs in each layer with
   concrete examples (which files move where, which stay)

2. **Full OptiX API scope** — define what "full OptiX API wrapper" means:
   - Which OptiX version to target (currently using OptiX 7.x)
   - Which API surface to wrap first: context creation, module/pipeline/program group
     setup, SBT management, `optixLaunch`, acceleration structure build
   - JVM API style: thin 1:1 bindings (like LWJGL) vs higher-level builder API
   - Estimate of work for full wrapping — likely Sprint 26+

3. **`BaseParams`/`MengerParams` extension pattern** — validate with a minimal prototype
   (two-file C++ program) that the struct extension compiles correctly and both cast
   patterns produce correct results before committing to the full rename

4. **Versioning strategy** — independent version numbers for optix-jni vs menger
   releases? Or kept in sync? How is the published version derived from the menger
   version?

5. **Maven Central requirements** — confirm Sonatype Central Portal account exists
   and GPG key is available; identify any blockers before the CI job is wired

**Output:** `docs/architecture/optix-jni-decoupling.md` (committed on this branch)

---

### Task 24.2: Package Rename `menger.optix` → `io.github.lene.optix`

**Estimate:** 4h

**Scala side:**
- Update `package` declarations in all 8 Scala files in `optix-jni/src/main/scala/`
- Update `import menger.optix.*` in any callers in `menger-app` and `menger-common`
- Update test files

**C++ side (JNIBindings.cpp):**
- Rename all ~80 JNI function symbols:
  `Java_menger_optix_*` → `Java_io_github_lene_optix_*`
- Update `FindClass("menger/optix/Light")` → `FindClass("io/github/lene/optix/Light")`
- Any other hardcoded class path strings

**Build files:**
- `optix-jni/build.sbt`: update `organization` from `io.github.lilacashes` → `io.github.lene`
  and confirm `name := "optix-jni"` stays
- Update any references in `.gitlab-ci.yml` or scripts

**Validation:** `sbt compile` clean; all tests pass.

---

### Task 24.3: Rename `Params` → `BaseParams`

**Estimate:** 3h

In `optix-jni/src/main/native/include/OptiXData.h`:
- Rename `struct Params` → `struct BaseParams`
- Remove 4D-specific fields (deferred to Task 24.5):
  `menger4d_data`, `num_menger4d`, `sierpinski4d_data`, `num_sierpinski4d`,
  `hexadecachoron4d_data`, `num_hexadecachoron4d`
- Remove caustics-specific fields (moved to `MengerParams` in Task 24.5)

Update all files that reference `Params`:
- `OptiXWrapper.cpp`, `OptiXWrapper.h`
- `PipelineManager.cpp`, `PipelineManager.h`
- `SceneParameters.cpp`, `SceneParameters.h`
- All base shaders: `optix_shaders.cu`, `hit_sphere.cu`, `hit_cylinder.cu`,
  `hit_cone.cu`, `hit_plane.cu`, `hit_triangle.cu`, `miss_plane.cu`, `helpers.cu`
- `JNIBindings.cpp`

**Validation:** `sbt "project optixJni" compile`; C++ Google Test suite passes.

---

### Task 24.4: Create `menger-geometry` sbt Subproject

**Estimate:** 2h

Create the module scaffold:

```
menger-geometry/
  build.sbt
  src/
    main/
      scala/menger/geometry/   (empty, placeholder)
      native/
        CMakeLists.txt          (stub — links against optix-jni native)
        include/
          MengerParams.h        (stub — populated in Task 24.5)
```

Wire into `build.sbt` (root):
```scala
lazy val mengerGeometry = project
  .in(file("menger-geometry"))
  .dependsOn(optixJni)
  .enablePlugins(JniNative)

lazy val mengerApp = project
  .in(file("menger-app"))
  .dependsOn(mengerGeometry, mengerCommon)  // optixJni now indirect
  ...
```

**Validation:** `sbt compile` succeeds for all modules (menger-geometry compiles empty).

---

### Task 24.5: Define `MengerParams` and Validate CUDA Compilation

**Estimate:** 3h

In `menger-geometry/src/main/native/include/MengerParams.h`:

```cpp
#pragma once
#include <optix_jni/BaseParams.h>   // include from optix-jni headers
#include "Menger4DData.h"
#include "Sierpinski4DData.h"
#include "Hexadecachoron4DData.h"
#include "CausticsParams.h"         // moved from optix-jni

struct MengerParams {
    BaseParams base;                          // MUST be first — layout extension

    // 4D geometry buffers
    Menger4DData*         menger4d_data;
    unsigned int          num_menger4d;
    Sierpinski4DData*     sierpinski4d_data;
    unsigned int          num_sierpinski4d;
    Hexadecachoron4DData* hexadecachoron4d_data;
    unsigned int          num_hexadecachoron4d;

    // Caustics
    CausticsParams caustics;
};
```

Write a minimal stub CUDA shader that casts to `MengerParams*` and reads a field —
verify it compiles with `nvcc`. This validates the pattern before Sprint 25 moves
real shaders.

**Validation:** Stub shader compiles; `sbt "project mengerGeometry" nativeCompile` succeeds.

---

### Task 24.6: Publication Setup — `menger-common`

**Estimate:** 2h

Add to `menger-common/build.sbt`:
- `organization := "io.github.lene"`
- `publishTo` pointing to GitLab Package Registry (env var `CI_JOB_TOKEN`)
- `publishTo` for Maven Central via sbt-sonatype or sbt-ci-release
- POM metadata: `description`, `homepage`, `licenses`, `scmInfo`, `developers`
- `publish / skip := false` (currently likely skipped)

Add CI jobs to `.gitlab-ci.yml`:
- `PublishCommon` — runs on tag pipeline, publishes `menger-common` to both registries

---

### Task 24.7: Publication Setup — `optix-jni`

**Estimate:** 3h

`optix-jni/build.sbt` already has partial metadata. Complete:
- Update `organization` to `io.github.lene` (matches 24.2)
- Add `sonatypeCredentialHost` / `sonatypeRepository` for Central Portal
- Add GPG signing plugin (`sbt-pgp`)
- Native artifact inclusion: the `.so` and `.ptx` must be bundled in the published JAR
  (verify `nativeCompile` output is included as managed resources — already done for
  local use, confirm it survives packaging for publication)
- Add CI job `PublishOptixJni` — runs on tag pipeline

**Validation:** `sbt "project optixJni" publishLocal` succeeds and JAR contains `.so`.

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 24.1 | Architecture review + design doc | 4h |
| 24.2 | Package rename `menger.optix` → `io.github.lene.optix` | 4h |
| 24.3 | Rename `Params` → `BaseParams`, remove 4D/caustics fields | 3h |
| 24.4 | Create `menger-geometry` sbt subproject skeleton | 2h |
| 24.5 | Define `MengerParams` struct + validate CUDA compilation | 3h |
| 24.6 | Publish setup: `menger-common` | 2h |
| 24.7 | Publish setup: `optix-jni` | 3h |
| **Total** | | **~21h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing (pre-push hook green)
- [ ] `sbt "project optixJni" publishLocal` and `sbt "project mengerCommon" publishLocal` succeed
- [ ] Design document committed at `docs/architecture/optix-jni-decoupling.md`
- [ ] `menger-geometry` module compiles (even if empty of real logic)
- [ ] CHANGELOG.md updated

---

## Notes

### Native Artifact in Published JAR

The current build bundles the `.so` and `.ptx` via `resourceGenerators`. Verify the
published JAR includes them — users of the library need the native binary without a
separate build step. If the JAR is platform-specific (Linux x86_64 only for now),
document this clearly in the README and POM.

### Maven Central GPG Requirement

Maven Central requires GPG-signed artifacts. Confirm before starting 24.7 that:
1. A GPG key exists (run `gpg --list-keys`)
2. The key is published to a keyserver (`gpg --keyserver keyserver.ubuntu.com --send-keys <KEY_ID>`)
3. The passphrase is available as a CI secret variable

### Full OptiX API Wrapping

Scoped in Task 24.1. Implementation starts in Sprint 25 (Task 25.4) with the core
functions; further wrapping is Sprint 26+. Do not attempt full wrapping in Sprint 24.
