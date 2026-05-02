# Code Quality Improvements — Open Issues

**Last Updated:** 2026-05-02

Resolved items are removed from this file entirely — git history is the record of what was fixed.

---

## High Priority


---


## Medium Priority

## Low Priority

### L-mesh4d-plan-duplication — TesseractMesh / TesseractSpongeMesh / TesseractSponge2Mesh constructed twice
**Category:** `DUPLICATION`
**Location:** `menger-app/src/main/scala/menger/engines/scene/MeshFactory.scala:73-114` and `:142-165`
**Est. Effort:** 0.5h
`MeshFactory.create` and `MeshFactory.gpu4DPlan` each switch on the same three 4D object types
(`tesseract`, `tesseract-sponge`, `tesseract-sponge-2`) and construct the same `Mesh4DProjection`
subclass with identical argument lists. Adding a fourth 4D-projected type means editing both
match expressions. Extract a private helper `mesh4DProjection(spec): Option[Mesh4DProjection]`
that returns the projection object (None for non-4D specs); have `create` call
`.toTriangleMesh` on it and `gpu4DPlan` call `Mesh4DGpuFlatten.quadsBuffer(p.mesh4D)` on it.

---

### L-tesseract-sponge-2-containment — level-2 faces straddle removed sub-cubes
**Location:** `menger-app/src/main/scala/menger/objects/higher_d/TesseractSponge2.scala`
**Est. Effort:** 4–8h
At level 2, 768 faces (out of 6144) have at least one corner inside a level-1 sub-cube that
should be removed (count-of-middle-indices ≥ 2). The number `768 = 16 × 48` is suspiciously
systemic, suggesting a construction flaw in `generateFlatParts` / `generatePerpendicularParts`
or in how `cornerPoints` is derived from parent faces that themselves straddle a sub-cube
boundary. This is **visually benign** — the surface renders cleanly after the 2026-04-24
normals fix (see `docs/superpowers/investigations/2026-04-21-tesseract-sponge-2-flap.md`) —
but the mesh is still not a strict boundary of the 4D Menger solid. Related: 1920 boundary
edges and 2112 triple-shared edges in the level-2 manifold histogram. A regression guard
(`TesseractSponge2MeshSpec`: "have at most 2000 boundary edges at level 2") is already active;
investigate the containment root cause in `TesseractSponge2.scala`.

---

### L-caustics-duplicate-config — CausticsConfig and dsl.Caustics duplicate fields and validation
**Location:** `optix-jni/.../RenderConfig.scala` and `menger-app/.../dsl/Caustics.scala`
**Est. Effort:** 0.5h
Both case classes carry identical fields (`photonsPerIteration`, `iterations`, `initialRadius`,
`alpha`) and the same `require` guards. The `Caustics.toCausticsConfig` bridge means any change
to limits must be made in two places. Consider extracting validation constants to a shared object
or letting the DSL type own the constraints and stripping them from `CausticsConfig`.

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
