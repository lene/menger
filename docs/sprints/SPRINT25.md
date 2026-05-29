# Sprint 25: optix-jni Implementation

**Sprint:** 25 - optix-jni Decoupling: Implementation
**Status:** Not Started
**Estimate:** ~28 hours
**Branch:** `feature/sprint-25`
**Dependencies:** Sprint 24 (architecture, BaseParams/MengerParams, module scaffold, publication setup)

---

## Goal

Complete the decoupling of `optix-jni` from Menger-specific code by moving all 4D
geometry and caustics into `menger-geometry`. Begin full OptiX API wrapping (scope
defined in Sprint 24, Task 24.1). Deliver a fully published `io.github.lene:optix-jni`
artifact with no Menger-specific types.

---

## Success Criteria

- [ ] All 4D geometry C++ (`hit_menger4d.cu`, `hit_sierpinski4d.cu`, `hit_hexadecachoron4d.cu`,
  `Project4D`, recursive IAS sponge, all 4D data structs) in `menger-geometry`
- [ ] `CausticsRenderer` in `menger-geometry`
- [ ] `optix-jni` contains zero Menger-specific types or geometry
- [ ] `menger-app` depends on `menger-geometry`; 4D API calls route through it
- [ ] Core OptiX API functions wrapped (scope per Sprint 24 design doc)
- [ ] All integration tests pass
- [ ] Published artifact works: a minimal external project can depend on `io.github.lene:optix-jni`
  and render a sphere without menger-app code

---

## Tasks

### Task 25.1: Move 4D C++ to `menger-geometry`

**Estimate:** 6h
**Depends on:** 24.3 (BaseParams), 24.5 (MengerParams defined)

Move to `menger-geometry/src/main/native/`:
- `shaders/hit_menger4d.cu`
- `shaders/hit_sierpinski4d.cu`
- `shaders/hit_hexadecachoron4d.cu`
- `include/Project4D.h`, `project4d.cu`
- Recursive IAS sponge logic from `OptiXWrapper.cpp`/`OptiXWrapper.h`
- Data structs: `Menger4DData`, `Sierpinski4DData`, `Hexadecachoron4DData`
  (move from `OptiXData.h` to `menger-geometry/include/`)

Update `menger-geometry/CMakeLists.txt` to compile these shaders to PTX and include
them as managed resources.

Update `menger-geometry` pipeline setup to register the 4D hit programs alongside
the base programs from `optix-jni`.

**Validation:** `sbt "project mengerGeometry" nativeCompile` succeeds; 4D PTX files present.

---

### Task 25.2: Move `CausticsRenderer` to `menger-geometry`

**Estimate:** 3h
**Depends on:** 24.5

Move to `menger-geometry/src/main/native/`:
- `CausticsRenderer.cpp`, `include/CausticsRenderer.h`
- `shaders/caustics_ppm.cu`
- `CausticsParams` struct (already part of `MengerParams` per 24.5)

Remove from `optix-jni`. Confirm `optix-jni` no longer references caustics types.

**Validation:** `sbt "project optixJni" compile` with no caustics references; caustics
integration tests still pass via `menger-geometry`.

---

### Task 25.3: Move 4D + Caustics Scala/JNI to `menger-geometry`

**Estimate:** 4h
**Depends on:** 25.1, 25.2

Move JNI bindings and Scala API for all 4D geometry and caustics:
- From `JNIBindings.cpp` (optix-jni): extract 4D + caustics JNI functions
  (`addMenger4DInstanceNative`, `updateMenger4DProjectionNative`,
  `addSierpinskiInstanceNative`, `addHexadecachoronInstanceNative`,
  `addRecursiveIASSpongeInstanceNative`, caustics JNI functions)
  → move to `menger-geometry/src/main/native/MengerJNIBindings.cpp`
- Scala: move relevant `@native` declarations and wrapper methods out of `OptiXRenderer`
  into a new `menger-geometry` Scala class (e.g. `MengerRenderer extends OptiXRenderer`)

**Validation:** `sbt compile` full build; `menger-app` uses `MengerRenderer` for 4D calls.

---

### Task 25.4: Begin Full OptiX API Wrapping

**Estimate:** 8h
**Depends on:** 24.1 (scope defined), 24.2 (package rename)

Implement the first tranche of OptiX API wrapping as defined in the Sprint 24 design
document. Expected scope (confirm against 24.1 output):

- `OptixDeviceContext` lifecycle: create, destroy, set log callback
- `OptixModule` creation from PTX/optixir source
- `OptixProgramGroup` creation (raygen, miss, hitgroup)
- `OptixPipeline` creation and linking
- `OptixShaderBindingTable` setup utilities
- `optixLaunch` wrapper

Each wrapped function gets:
- A C++ JNI binding in `JNIBindings.cpp`
- A Scala `@native` declaration + thin wrapper
- A unit test (where testable without GPU — use mocking or skip with `assumeTrue(gpuAvailable)`)

**Note:** Do not attempt to wrap the full API surface in this sprint. Cover the core
launch path; leave denoiser, curves primitives, and motion blur for a later sprint.

---

### Task 25.5: Update `menger-app` Dependencies

**Estimate:** 2h
**Depends on:** 25.3

- `menger-app` `dependsOn` changes: add `mengerGeometry`, keep `mengerCommon`
- Remove any direct `optixJni` project reference from `mengerApp` if now fully indirect
- Confirm all scene configurators that call 4D API now go through `MengerRenderer`
- Update `javaOptions` library path to include `menger-geometry` native output

**Validation:** `sbt "project mengerApp" compile`; `sbt run` renders a scene.

---

### Task 25.6: Tests & Integration Validation

**Estimate:** 3h

- Run full pre-push hook end-to-end
- Confirm all integration tests pass (181 scenarios)
- Add one smoke test in `menger-geometry`: init `MengerRenderer`, add a Menger4D
  instance, render 1 frame — verifies the full stack from Scala to CUDA

---

### Task 25.7: Documentation

**Estimate:** 2h

- `optix-jni/README.md`: standalone library usage — how an external project adds the
  dependency, loads the native library, initialises the renderer, renders a sphere
- `menger-geometry/README.md`: brief note that this is an in-repo Menger-specific
  extension of optix-jni, not intended for external use
- Update `docs/arc42/09-architectural-decisions.md` with the 3-layer decision

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 25.1 | Move 4D C++ to menger-geometry | 6h |
| 25.2 | Move CausticsRenderer to menger-geometry | 3h |
| 25.3 | Move 4D + caustics Scala/JNI to menger-geometry | 4h |
| 25.4 | Begin full OptiX API wrapping (core launch path) | 8h |
| 25.5 | Update menger-app dependencies | 2h |
| 25.6 | Tests + integration validation | 3h |
| 25.7 | Documentation | 2h |
| **Total** | | **~28h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green (all tests, coverage, integration)
- [ ] `optix-jni` contains no Menger-specific types
- [ ] `menger-geometry` compiles and renders 4D scenes correctly
- [ ] Published JAR verified: external `build.sbt` can depend on and use `optix-jni`
- [ ] CHANGELOG.md updated
- [ ] arc42 section 9 updated with 3-layer architecture decision

---

## Notes

### Pipeline Registration in menger-geometry

The OptiX pipeline in `optix-jni` registers hit programs from its own PTX files.
`menger-geometry` must extend this registration to include its 4D shaders. Two
approaches:
1. `menger-geometry` owns the full pipeline setup (calls `PipelineManager` with both
   base + 4D PTX files) — simpler, but menger-geometry controls pipeline
2. `optix-jni` exposes a `registerHitProgram(ptxFile, entryPoint)` API that
   menger-geometry calls — cleaner boundary but more API surface to design

Decide which approach based on Sprint 24 design doc (Task 24.1).

### External Validation

Task 25.6 success criterion: a minimal `build.sbt` outside the menger repo:
```scala
libraryDependencies += "io.github.lene" %% "optix-jni" % "0.7.1"
```
can compile and call `new OptiXRenderer().initialize(...)`. This is the real test
that the publication works end-to-end.
