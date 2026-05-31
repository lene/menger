# Sprint 26: Repository Split & Code Health

**Sprint:** 26 - Repository Split & Code Health
**Status:** Not Started
**Estimate:** ~28 hours
**Branch:** `feature/sprint-26`
**Dependencies:** Sprint 25 (published optix-jni and menger-common artifacts must exist)

---

## Goal

Split the published artifacts (menger-common, optix-jni) into separate repositories now
that they are independent libraries. Fix all open medium-priority CODE_IMPROVEMENTS
items. Remove the obsolete legacy CPU 4D path.

---

## Success Criteria

- [ ] `optix-jni` and `menger-common` published to registry; external project verified
- [ ] `menger-common` and `optix-jni` live in separate repositories, published independently
- [ ] `menger` repo retains only `menger-geometry` and `menger-app`
- [ ] CI/CD updated for cross-repo dependency resolution
- [ ] All 6 CODE_IMPROVEMENTS medium-priority items resolved and removed from CODE_IMPROVEMENTS.md
- [ ] `Mesh4D` and `RotatedProjection` deleted
- [ ] All tests pass

---

## Tasks

### Task 26.0: Publish optix-jni and menger-common

**Estimate:** 3h
**Depends on:** Sprint 25 (3-layer architecture complete, CI green on 0.7.2 tag)

The publication CI jobs (`PublishOptixJni`, `PublishCommon`) were wired up in
Sprint 24.7 but never triggered for a real release. This task verifies the
full publish path end-to-end.

**Steps:**
1. Trigger the 0.7.2 tag pipeline on GitLab and confirm `PublishOptixJni` and
   `PublishCommon` jobs complete successfully.
2. Verify artifacts appear in the GitLab Package Registry and/or Maven Central
   (depending on which registry the CI is configured to publish to).
3. Create a minimal external `build.sbt` outside the menger repo:
   ```scala
   libraryDependencies += "io.github.lene" %% "optix-jni" % "0.1.0"
   ```
   Verify it compiles and `new OptiXRenderer().initialize()` links correctly.
4. If any publish step fails, fix the CI/signing/credential configuration before
   proceeding to 26.1.

**Validation:** External project compiles against the published artifact.

---

### Task 26.0a: optix-jni API Documentation

**Estimate:** 2h
**Depends on:** 26.0 (publish path confirmed working)

Add Scaladoc to all public API types in `optix-jni`:
- `OptiXRenderer` — lifecycle (`initialize`, `dispose`), render loop
- `NativeOptiXApi` — handle semantics, 0L failure convention, ownership model
- All five API traits (`OptiXSphereApi`, `OptiXMeshApi`, etc.) — what each method
  does, parameter units, error conditions
- `RenderResult`, `RenderHealth`, `Material` — field semantics

Also update `optix-jni/README.md` with: minimum JVM flags (`-Djava.library.path`),
sbt / Maven / Gradle dependency snippet, GPU/driver requirements.

**Validation:** `sbt doc` produces Scaladoc with no missing-doc warnings on public API.

---

### Task 26.0b: optix-jni Non-GPU Unit Tests

**Estimate:** 2h
**Depends on:** none (pure Scala, no GPU needed)

Add unit tests that run without a GPU (all guarded with `assume(OptiXRenderer.isLibraryLoaded, ...)`):
- Library loading: `isLibraryLoaded` returns non-exception result
- Handle safety: `destroyContext(0L)`, `destroyModule(0L, 0L)`, `destroyPipeline(0L, 0L)` do not crash
- `createContext()` → `0L` path handled gracefully by `NativeOptiXApi`
- `OptiXRenderer.initialize()` idempotence: calling twice returns same result

Existing `NativeOptiXApiTest` already covers some of this — move and expand it
into a proper `optix-jni` test module (not `menger-app`).

**Validation:** Tests run in CI without GPU (`Test:Full` job); 0 failures.

---

### Task 26.1: Repository Split

**Estimate:** 8h

Extract `menger-common` and `optix-jni` into separate GitLab/GitHub repositories.

**Steps:**
1. Create new repos: `github.com/lene/menger-common`, `github.com/lene/optix-jni`
2. Use `git filter-repo` to extract history for each module
3. Set up CI/CD in each new repo (compile, test, publish on tag)
4. Update `menger` root `build.sbt`: replace project references with published artifact
   dependencies for `menger-common` and `optix-jni`
5. Update `.gitlab-ci.yml` and GitHub Actions in the new repos
6. Archive the `menger-common/` and `optix-jni/` directories in the menger repo
   (or remove entirely once CI is green)

**Validation:** `sbt compile` in menger repo resolves menger-common and optix-jni from
registry, not from local project references.

---

### Task 26.2: Fix M-cuda-gas-buffer-leak

**Estimate:** 2h

`d_gas_output_buffer` is allocated with `cudaMalloc` before `OPTIX_CHECK`. If
`OPTIX_CHECK` throws, the buffer is leaked.

Fix: use RAII wrapper (`CudaBuffer` if available) or add explicit cleanup in the
catch block.

---

### Task 26.3: Fix M-cuda-texture-array-leak

**Estimate:** 2h

`cudaArray_t` is created before `cudaCreateTextureObject`. If texture object creation
fails, the array is leaked.

Fix: RAII pattern or explicit `cudaFreeArray` in error path.

---

### Task 26.4: Fix M-render-null-type-contract

**Estimate:** 1h

`renderWithStats` is typed `RenderResult` but can return null in error paths (already
partially fixed in Sprint 23.9). Audit all code paths; ensure return type is
`Option[RenderResult]` throughout.

---

### Task 26.5: Fix M-texture-index-overloading

**Estimate:** 2h

`texture_index` in `InstanceMaterial` is used for both geometry data indexing and
image texture indexing — two incompatible semantics. Rename to disambiguate:
- `geometry_data_index` for geometry buffer lookup
- `image_texture_index` for texture array lookup

Update all shaders and C++ code that reads this field.

---

### Task 26.6: Fix M-arch-config-naming

**Estimate:** 1h

The `*Config` ArchUnit naming rule cannot be enabled because some Config types live
outside the `menger.common.config` package. Move them to the correct package and
enable the rule.

---

### Task 26.7: Remove Legacy CPU 4D Path

**Estimate:** 2h

Delete `Mesh4D`, `RotatedProjection`, and any other CPU-side 4D projection code that
has been superseded by the GPU path (Sprint 18+). Verify no callers remain before
deleting.

---

### Task 26.8: Test LibGDX Wrapper Paths

**Estimate:** 3h

`SceneConfigurator`, `OptiXRendererWrapper`, and `CameraState` (all in
`io.github.lene.optix` under `menger-app`) are 0–17% covered because they are
only instantiated inside the LibGDX `ApplicationListener` lifecycle, which
requires a live display. These paths are responsible for connecting OptiX
rendering to the interactive engine — exactly the kind of glue code that hides
erratic bugs.

**Approach:**
- Evaluate LibGDX headless backend (`HeadlessApplication`) for unit test use
- If headless backend is viable: add unit tests for `SceneConfigurator` scene
  setup, `OptiXRendererWrapper` init/render/dispose lifecycle, `CameraState`
  update logic
- If not viable: extract testable logic from each class behind interfaces so
  it can be exercised without a running `Application`

**Validation:** `SceneConfigurator`, `OptiXRendererWrapper`, `CameraState`
each reach ≥70% statement coverage. Remove `L-libgdx-wrapper-untested` from
`CODE_IMPROVEMENTS.md`.

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 26.0 | Publish optix-jni + menger-common; verify external project | 3h |
| 26.0a | optix-jni API documentation (Scaladoc + README) | 2h |
| 26.0b | optix-jni non-GPU unit tests | 2h |
| 26.1 | Repository split | 8h |
| 26.2 | Fix CUDA GAS buffer leak | 2h |
| 26.3 | Fix CUDA texture array leak | 2h |
| 26.4 | Fix renderWithStats null contract | 1h |
| 26.5 | Fix texture_index semantic overloading | 2h |
| 26.6 | Fix Config naming rule | 1h |
| 26.7 | Remove legacy CPU 4D path | 2h |
| 26.8 | Test LibGDX wrapper paths | 3h |
| **Total** | | **~24h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] CODE_IMPROVEMENTS.md: all 5 medium-priority items removed
- [ ] CHANGELOG.md updated
