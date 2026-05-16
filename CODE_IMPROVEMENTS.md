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

### Issue M-objectspec-god-class: LARGE_CLASS / DATA_CLUMPS — ObjectSpec is a 528-line god class

**Location**: `menger-app/src/main/scala/menger/ObjectSpec.scala`
**Impact**: Medium
**Debt Cost**: Moderate (1–3 days)

~30-field case class plus a 297-line monolithic `parse()` (parse entry at line 145, helper parsers run to ~441). The same primitive groups repeat throughout the class and parser as data clumps:

- position `(x, y, z)`
- rotation 3D `(rotX, rotY, rotZ)`
- rotation 4D `(rotXW, rotYW, rotZW)` + `projection4D`
- material — appears 3× (main, edges, color2)
- cone/plane geometry `(apex, base, radius, normal, distance, checkerSize)`
- procedural `(proceduralType, proceduralScale)`
- texture maps `(texture, normalMap, roughnessMap)`

`parse()` is one long for-comprehension chaining 20+ validators with no typed intermediate stages, so an early failure blocks all later checks.

**Recommendation**: Extract value objects (`Position3D`, `Rotation3D`, `Rotation4D`/`ProjectionSpec`, `MaterialSpec`, `ProceduralSpec`, `TextureMaps`) and split `parse()` into per-aspect parsers composed at the top level. Supersedes the previously-closed `L-objectspec-parse-length` with broader scope (the parser remains a monolith).

---


### Issue M-interactiveengine-state-dup: DUPLICATION / LONG_METHOD — InteractiveEngine duplicate 4D state + repeated warning logic

**Location**: `menger-app/src/main/scala/menger/engines/InteractiveEngine.scala`
**Impact**: Medium
**Debt Cost**: Minor–Moderate

`anim4DState` (line 91) and `cpu4DState` (line 98) are two `AtomicReference[Option[WithAnimation.Anim4DState]]` with identical structure, differing only GPU vs CPU. `warnIfHighLevel()` (line 225, ~51 L) repeats the same warning body across geometry types via match arms; `handleEvent()` (line 116, ~41 L) and `buildScene4DTrackedOrFallback()` (~44 L) are long with GPU/CPU branching.

**Recommendation**: Merge the two references into a single `Scene4DCache(gpu, cpu)`; replace the repeated warning match arms with a data-driven `LevelWarningConfig` registry.

---


## Low Priority


### Issue L-optixrenderer-size: LARGE_CLASS — OptiXRenderer at 995 lines, ~65 public methods

**Location**: `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala`
**Impact**: Low
**Debt Cost**: Major (split across several sprints)

Re-measured 2026-05-16: 995 lines, ~65 public methods, 2 long methods (both justified Scala→C++ marshalling). Single class wraps the entire OptiX JNI surface: geometry upload, instance management, camera, lights, render params, texture upload, stats, caustics. Growth since last review is organic — no anomalous increase.

**Recommendation**: Consider splitting by concern into focused facades (e.g. `SphereRenderer`, `MeshRenderer`, `PlaneRenderer`, `TextureFacade`, `IASInstanceManager`) that delegate to the underlying JNI class. The JNI native methods can stay in one class; the Scala API is where grouping helps. Non-blocking — revisit only if the public method count exceeds ~80.

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
