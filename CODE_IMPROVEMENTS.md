# Code Quality Improvements — Open Issues

**Last Updated:** 2026-05-16

Resolved items are removed from this file entirely — git history is the record of what was fixed.

---

## High Priority


---


## Medium Priority

### Issue M-texture-builder-gap: MISSING FEATURE — Cone and plane shaders don't support PBR map textures

**Location**: `optix-jni/src/main/native/shaders/hit_cone.cu`, `hit_plane.cu`
**Impact**: Medium
**Debt Cost**: Moderate (1–2 days per geometry type: UV generation + shader sampling)

Procedural textures now work for both cone and plane (fixed in refactor commit). Image textures and PBR maps (normal map, roughness map) are still missing because `texture_index` is repurposed on both geometry types to index geometry data arrays (`cone_data` / `plane_data`), leaving no slot for image texture lookup. `applyNormalMap` / `applyRoughnessMap` also require UV coordinates which are not generated for these geometries.

Adding image + PBR map support requires:
1. UV coordinate generation in the hit shader (planar XZ for plane, cylindrical for cone)
2. A separate image-texture index field (since `texture_index` is taken by geometry data)
3. Calling `sampleInstanceTexture`, `applyNormalMap`, `applyRoughnessMap` with generated UVs

**Recommendation**: Implement per geometry type when needed. Follow `hit_sphere.cu` as reference.

---

## Low Priority

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
