# Code Quality Review — 2026-06-26

Sprint 29 close review. 64 files changed (+1163/-355), 31 commits. Two new features (denoiser, curves) integrated across JNI → DSL → CLI → engine pipeline.

## Summary

The sprint is structurally clean — denoiser lifecycle is consistent across all five engines, curve DSL follows the SceneObject pattern, and test coverage is thorough for the happy path. Three gaps warrant attention: animation engines never re-apply denoise/accumulation settings per-frame, the CLI `--object` path hardcodes accumulation to 1 with no user-facing flag, and the Curve builder silently discards texture/rotation fields carried by its DSL type. None are release-blocking for v0.7.6.

## Tooling status

- Scalafix: could not verify (local environment lacks published optix-jni jar for menger-geometry's extractOptixJniNativeApi). CI runs it — rely on pipeline.
- WartRemover: runs as part of compile.
- ArchUnit: 14+5 active rules in ArchitectureSpec + ArchitecturePhase2Spec.
- clang-tidy: configured, runs in pre-push + CI.
- cppcheck: configured, runs in pre-push + CI.
- Tooling gaps: none introduced this sprint.

## Findings

### 1. Denoise/accumulation settings not re-applied per-frame in animation engines

**Where**: `WithAnimation.scala:104`, `WithPreview.scala:112`, `CliAnimationEngine.scala:79`
**Impact**: High — denoise/accumulation are set once at engine `create()` time; per-frame `SceneConfigs` updates (e.g. varying `RenderSettings.denoise` or `accumulation`) are silently ignored during animation.
**Effort**: hours

**What**: Sprint 29 consolidated denoise + accumulation into `BaseEngine.configureOutputMode()`, called from `create()`. But the per-frame `render()` paths in `WithAnimation`, `WithPreview`, and `CliAnimationEngine` never invoke it, despite computing new `SceneConfigs` per frame that carry `denoiseMode` and `accumulationFrames` fields.

**Why it matters**: If a DSL animation scene varies these settings between frames, the behavior is silently incorrect. More acutely, a scene that sets `accumulation = 2` in `RenderSettings` and relies on `--denoise` will work in interactive mode but not in animation mode.

**Suggested direction**: Call `configureOutputMode(configs)` at the top of each engine's per-frame `render()` method, mirroring the `setRenderConfig(configs.renderConfig)` pattern already there.

### 2. CLI `--object` path hardcodes accumulation to 1 — no `--accumulation-frames` flag

**Where**: `Main.scala:211-212`, `MengerCLIOptions.scala:395-408`
**Impact**: Medium — the `--denoise` flag's help text promises "denoise the final accumulated frame", but CLI `--object` users have no way to set accumulation count. The denoiser runs on a single noisy frame.
**Effort**: hours

**What**: The DSL path threads `RenderSettings.accumulation` through `SceneConfigs` → `OptiXEngineConfig` → `BaseEngine`. The CLI `--object` path in `Main.scala` hardcodes `accumulationFrames = 1`. `MengerCLIOptions` has no `--accumulation` or `--accumulation-frames` option.

**Suggested direction**: Add `--accumulation-frames <int>` (or `--accumulation <int>`) to `MengerCLIOptions`, thread it through the `--object` path in `Main.scala`, and validate `≥ 1`.

### 3. Curve DSL type carries texture/rotation fields that CurveSceneBuilder silently ignores

**Where**: `CurveSceneBuilder.scala:22-30`, `SceneObject.scala:428-462`
**Impact**: Medium — the `Curve` DSL type inherits `SceneObject` fields (texture, videoTexture, normalMap, roughnessMap, proceduralType, proceduralScale, rotation) via `baseObjectSpec()`, but the builder never reads them. Users who set `texture = "wood.png"` on a Curve will see no effect and no warning.
**Effort**: days

**What**: Every other analytical-primitive builder (`SphereSceneBuilder`, `ConeSceneBuilder`) applies textures and transformations. `CurveSceneBuilder.buildScene` reads only `spec.curveData` — texture loading (`TextureManager.loadTextures`), `applyInstanceTextures`, and transform application are all absent. The builder also silently ignores `Curve.pos`/`Curve.size` encoded into `ObjectSpec.x/y/z/size`.

**Why it matters**: The API surface is misleading — DSL fields that compile and appear valid produce no runtime effect. This is especially confusing because the OptiX curves primitive may genuinely not support UV-mapped textures or per-instance transforms, but the DSL type should either not expose those fields or document the limitation.

**Suggested direction**: Either (a) override `baseObjectSpec` in `Curve` to exclude unsupported fields, or (b) add explicit `require` checks rejecting them, or (c) implement texture/transform support if the OptiX primitive allows it. At minimum, document in the user guide.

### 4. `configureOutputMode` calls native methods without null-handle guard

**Where**: `BaseEngine.scala:33-35`, `OptiXRenderer.scala:269-271`
**Impact**: Medium — `setDenoisingEnabled` and `setAccumulationFrames` are `public native` with no Scala wrapper performing precondition checks. If called before `renderer.initialize()` or after `dispose()`, the result is a native SIGSEGV.
**Effort**: hours

**What**: All other instance-adding methods (e.g. `addCurveInstance`) route through `OptiXMeshApi` trait default methods that `require()` array validity. `setDenoisingEnabled`/`setAccumulationFrames` have no equivalent guard. In practice, engine callers always initialize first, but there is no enforced ordering — a future refactor could violate this silently.

**Suggested direction**: Add Scala wrapper methods that check `isInitialized` (or `nativeHandle != 0L`) before calling the native method, matching the guard pattern used in `OptiXRenderApi` for render calls.

### 5. `CurveSceneBuilder.buildScene` error path has no test coverage

**Where**: `CurveSceneBuilder.scala:26-28`, `CurveSuite.scala` (no mock coverage)
**Impact**: Low — the GPU-failure path (`addCurveInstance` returning `-1`) is wrapped in `Try {}` but never exercised in tests. The integration test covers only the happy path.
**Effort**: hours

**Suggested direction**: Add a scalamock test in `CurveSuite` or a new `CurveSceneBuilderSuite` that mocks the renderer to return `-1` from `addCurveInstance` and asserts the `Try`→`Failure` path.

### 6. `SceneType.Curves` classification exists in `RenderModeSelector` but has no matching dispatch branch in `BaseEngine`

**Where**: `RenderModeSelector.scala:23-24`, `BaseEngine.scala:91,116`
**Impact**: Low — today Curves fall through to generic dispatch in `BaseEngine` which resolves correctly. But `SceneType.TriangleMeshes` has explicit branches; Curves asymmetry means any pre-processing logic added to one won't reach the other.
**Effort**: hours

**Suggested direction**: Either add explicit `case SceneType.Curves(specs) =>` branches alongside the `TriangleMeshes` branches in `BaseEngine.buildSceneFromSpecs`/`buildSceneFromConfigs`, or remove the `SceneType.Curves` variant and let curves route purely through analytical-primitive dispatch.

## Carried forward from prior review

- `L-upload-texture-file-raw-int` — still deferred (fail-fast vs graceful-skip decision)
- `L-project4d-async-error` — still deferred (contract documentation)

## Resolved since last review

- `menger-geometry/build.sbt` OptixJniSource references — found during this review, fixed in `d5a78de6`.

## Positive patterns worth preserving

- Denoiser lifecycle in `configureOutputMode` — clean consolidation of a cross-cutting concern into one method, called once at engine creation, preserved across `clearAllInstances()` in render loops.
- `DenoiseMode` as a sealed enum with `Off | Final` — well-typed, appropriate scope, easy to extend.
- `Curve` DSL type validation — thorough NaN/Inf/finite checks on control points, correct closed-curve wrapping.

---

# Code Quality Improvements — Open Issues

**Last Updated:** 2026-06-26

Resolved items are removed from this file entirely — git history is the record of what was fixed.

---

## High Priority

*(none)*

---

## Medium Priority

*(all resolved — Sprint 30)*

---

## Low Priority

| ID | Issue | Location |
|----|-------|----------|
| L-upload-texture-file-raw-int | `uploadTextureFromFile` returns raw negative `Int` on failure while `uploadTexture` throws `TextureUploadException`. Behavior change needed: three production callers treat negative index as "skip and continue". Deferred until fail-fast vs graceful-skip decision is made. | `optix-jni/.../OptiXTextureApi.scala:67` |
| L-project4d-async-error | `cudaGetLastError()` at `project4d.cu:147` only captures launch-configuration errors, not async kernel execution errors. The required `cudaDeviceSynchronize` is in the caller, not the callee — document the contract or move the sync inside. | `menger-geometry/src/main/native/project4d.cu:147` |

---

## Tooling Gaps

*(all gaps identified in prior reviews have been closed)*

| Tool | Status | Where |
|------|--------|--------|
| ArchUnit | Closed — 14+6 active rules | `ArchitectureSpec.scala`, `ArchitecturePhase2Spec.scala` |
| cppcheck | Closed — pre-push + CI | `.cppcheck-suppress`, `.git_hooks/pre-push`, `.gitlab-ci.yml` |
| clang-tidy | Closed — pre-push + CI | `.clang-tidy`, `.git_hooks/pre-push`, `.gitlab-ci.yml` |

