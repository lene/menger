# Sprint 26: Repository Split & Code Health

**Sprint:** 26 - Repository Split & Code Health
**Status:** In Progress
**Estimate:** ~56 hours
**Branch:** `feature/sprint-26`
**Dependencies:** Sprint 25 (3-layer architecture complete; CI green on 0.7.2 tag)

---

## Goal

Split the published artifacts (menger-common, optix-jni) into separate repositories now
that they are independent libraries. Fix every open issue in CODE_IMPROVEMENTS.md,
including high-priority carried notes, medium-priority issues, and low-priority cleanup.
Remove the obsolete legacy CPU 4D path. Use GitLab Package Registry as the first
publication target for testing, then publish the standalone libraries to Maven Central
after the quality work is complete.

---

## Success Criteria

- [ ] `optix-jni` and `menger-common` published to GitLab Package Registry with a CI smoke version; external project verified
- [ ] `menger-common` and `optix-jni` live in separate repositories, published independently
- [ ] `menger-common` and `optix-jni` published to Maven Central after all quality fixes
- [ ] `menger` repo retains only `menger-geometry` and `menger-app`
- [ ] CI/CD updated for cross-repo dependency resolution
- [ ] All open High/Medium/Low Priority CODE_IMPROVEMENTS issues resolved and removed
- [ ] CODE_IMPROVEMENTS.md retains only Feature Ideas and Accepted/Deferred decisions
- [ ] `Mesh4D` and `RotatedProjection` deleted
- [ ] All tests pass

---

## Tasks

### Task 26.0: GitLab Package Registry Publish Smoke Test

**Estimate:** 3h
**Depends on:** Sprint 25 (3-layer architecture complete, CI green on 0.7.2 tag)

The publication CI jobs (`PublishOptixJni`, `PublishCommon`) are tied to real app
release tags. Do not create a dummy app release just to test publication. Instead,
verify the GitLab Package Registry path with a manual MR CI job that publishes
unique, pipeline-scoped smoke versions of `menger-common` and `optix-jni`, then
consumes them from a clean external sbt project.

**Steps:**
1. Run the manual `PublishLibraries:GitLabSmoke` job from the Sprint 26 MR pipeline.
2. Confirm the job publishes both libraries to the GitLab Package Registry with a
   version like `0.1.0-smoke.<pipeline-iid>.<job-id>`.
3. Confirm the same job creates a clean external sbt project outside the menger repo,
   resolves both smoke artifacts from the GitLab registry, compiles, and runs the
   verifier.
4. Confirm the verifier imports `menger.common.Color`, loads `OptiXRenderer` from the
   published `optix-jni` artifact, calls `initialize()`, and disposes the renderer.
5. If publication or consumption fails, fix the CI credentials, Maven metadata, native
   resource packaging, or resolver setup before proceeding to 26.1. Maven Central
   signing/release setup is intentionally deferred to 26.13 after the code-quality
   cleanup.

**Validation:** `PublishLibraries:GitLabSmoke` succeeds, and its external project
compiles and initializes `OptiXRenderer` using only published GitLab registry artifacts.

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
each reach ≥70% statement coverage. Remove `M-libgdx-wrapper-untested` from
`CODE_IMPROVEMENTS.md`.

---

### Task 26.9: Fix M-arch-archunit-case-class-field

**Estimate:** 3h

ArchUnit `haveOnlyFinalFields` reports false positives for Scala case class `val`
fields, and package checks treat synthetic `java.io.Serializable` inheritance as
domain file-IO usage.

Fix: replace the brittle final-field rule with a Scala-aware predicate that detects
mutable `var` fields, and exclude `java.io.Serializable` from file-IO package checks.

**Validation:** Re-enable the affected ignored ArchitecturePhase2Spec rules.

---

### Task 26.10: Fix M-arch-objects-logging

**Estimate:** 2h

`ParametricTessellator` and `higher_d/Rotation` use SLF4J from the pure geometry
layer.

Fix: remove direct logging from `menger.objects`. If callers need progress metadata,
return it from the computation and let outer layers log.

**Validation:** Re-enable the objects-logging ArchitecturePhase2Spec rule.

---

### Task 26.11: Fix M-objectspec-optix-coupling

**Estimate:** 8h

`ObjectSpec` imports `menger.optix.Material`, coupling core scene description to the
JNI layer.

Fix: move `Material` to `menger-common` or introduce a pure common material model that
`optix-jni` consumes. Update imports and verify no domain model requires native library
loading.

**Validation:** `ObjectSpec` and scene-building tests compile without depending on
`menger.optix`.

---

### Task 26.12: Resolve Low-Priority CODE_IMPROVEMENTS Sweep

**Estimate:** 10h

Fix every remaining open low-priority CODE_IMPROVEMENTS item rather than carrying a
second backlog into the split repositories:
- `L-upload-texture-file-raw-int`
- `L-m4d-level-const`
- `L-m4d-error-codes`
- `L-m4d-scene4d-sumtype`
- `L-m4d-builder-validation`
- `L-jni-release-mode-readonly`
- `L-cuda-synchronize-unchecked`
- `L-cuda-dispose-swallows`
- `L-tonemapping-magic-ints`

Also remove any stale carried note in the High Priority section after verifying the
pattern has not recurred.

**Validation:** CODE_IMPROVEMENTS.md has no High/Medium/Low Priority open issues.

---

### Task 26.13: Publish Libraries to Maven Central

**Estimate:** 5h
**Depends on:** 26.12 (all CODE_IMPROVEMENTS issues resolved), 26.1 (repositories split)

Publish the standalone `menger-common` and `optix-jni` libraries to Maven Central after
the repository split and quality cleanup are complete. GitLab Package Registry remains
the first test target; Central publication is the public distribution target once the
libraries are stable.

**Steps:**
1. Configure the split repositories for Sonatype Central Portal publication, including
   credentials, signing, metadata, and release automation.
2. Publish `io.github.lene:menger-common_3` and `io.github.lene:optix-jni_3` to
   Maven Central with their library versions.
3. Verify both artifacts appear on Maven Central and resolve without GitLab registry
   credentials.
4. Create a clean external project that depends on the Central artifacts and verifies
   `menger-common` compilation plus `OptiXRenderer` loading/initialization.

**Validation:** External project resolves both artifacts from Maven Central only.

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
| 26.9 | Fix ArchUnit Scala case-class false positives | 3h |
| 26.10 | Remove logging from pure geometry classes | 2h |
| 26.11 | Decouple ObjectSpec from optix Material | 8h |
| 26.12 | Resolve low-priority CODE_IMPROVEMENTS sweep | 10h |
| 26.13 | Publish libraries to Maven Central | 5h |
| **Total** | | **~56h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] Maven Central publication verified from a clean external project
- [ ] CODE_IMPROVEMENTS.md: no open High/Medium/Low Priority issues remain
- [ ] Any retained CODE_IMPROVEMENTS.md entry is either a Feature Idea or an
      Accepted/Deferred decision with an explicit rationale
- [ ] CHANGELOG.md updated
