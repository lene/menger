# Code Quality Improvements — Open Issues

**Last Updated:** 2026-05-16

Resolved items are removed from this file entirely — git history is the record of what was fixed.

---

## High Priority


---


## Medium Priority

### Issue M-texture-builder-gap: MISSING FEATURE — Cone and plane builders silently drop texture params

**Location**: `menger-app/src/main/scala/menger/engines/scene/ConeSceneBuilder.scala:23–37`, `PlaneSceneBuilder.scala:33–38`
**Impact**: Medium
**Debt Cost**: Trivial (30 min)

`ConeSceneBuilder.buildScene` never calls `setProceduralTexture`, `setMapTextures`, or `TextureManager`. `PlaneSceneBuilder` calls `setProceduralTexture` but skips `setMapTextures`. Any `normal-map`, `roughness-map`, `texture`, or `proc-type` param on a cone or normal-map on a plane is silently ignored — data loss with no error.

**Recommendation**: Apply the same texture wiring block used in `SphereSceneBuilder:55–62` to both builders. Also extract the repeated 6-line block (procedural + normalIdx + roughnessIdx) into a helper in `SceneBuilder` trait or `TextureManager` to eliminate the duplication across Sphere, TriangleMesh, Plane, and Cone builders.

---

### Issue M-texture-wiring-duplication: DUPLICATION — PBR texture wiring block copied across builders

**Location**: `SphereSceneBuilder.scala:57–62`, `TriangleMeshSceneBuilder.scala:123–128`
**Impact**: Medium
**Debt Cost**: Minor (2–3 h)

Identical 6-line block:
```scala
if spec.proceduralType != 0 then
  renderer.setProceduralTexture(id, spec.proceduralType, spec.proceduralScale)
val normalIdx    = spec.normalMap.flatMap(textureIndices.get).getOrElse(-1)
val roughnessIdx = spec.roughnessMap.flatMap(textureIndices.get).getOrElse(-1)
if normalIdx >= 0 || roughnessIdx >= 0 then
  renderer.setMapTextures(id, normalIdx, roughnessIdx)
```
Repeated verbatim in every builder that supports PBR maps. Adding a new map type (e.g. emissive map) requires editing every builder.

**Recommendation**: Extract to `SceneBuilder`:
```scala
protected def applyInstanceTextures(
  id: Int, spec: ObjectSpec,
  textureIndices: Map[String, Int],
  renderer: OptiXRenderer
): Unit = ...
```

---

## Low Priority

### Issue L-objectspec-parse-length: LONG_METHOD — ObjectSpec.parse() is 297 lines

**Location**: `menger-app/src/main/scala/menger/ObjectSpec.scala:145–441`
**Impact**: Low
**Debt Cost**: Minor (3–4 h)

The `parse()` for-comprehension is 297 lines. While each step delegates to a private helper, the chain is so long it's difficult to find where a specific field is parsed. New fields always go at the bottom, making the ordering arbitrary.

**Recommendation**: Group related parse steps into named intermediate helpers that return `Either[String, PartialSpec]` (e.g. `parseTextureParams`, `parsePBRParams`, `parse4DParams`), then compose those in the top-level `parse()`. No functional change — purely organizational.

---

### Issue L-optixrenderer-size: LARGE_CLASS — OptiXRenderer at 995 lines, 65 public methods

**Location**: `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala`
**Impact**: Low
**Debt Cost**: Major (split across several sprints)

Single class wraps the entire OptiX JNI surface: geometry upload, instance management, camera, lights, render params, texture upload, stats, caustics. Every new feature adds public methods. The class is already hard to navigate.

**Recommendation**: Consider splitting by concern into focused facades (e.g. `TextureFacade`, `InstanceFacade`, `RenderFacade`) that delegate to the underlying JNI class. The JNI native methods can stay in one class; the Scala API is where grouping helps. Low urgency — the JNI boundary is inherently monolithic — but worth tracking.

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
