# Code Quality Improvements — Open Issues

**Last Updated:** 2026-04-19 (Sprint 17 post-sprint review)

Resolved items are removed from this file entirely — git history is the record of what was fixed.

---

## High Priority

### H-tesseract-sponge-dark — TesseractSponge2 renders as solid dark cube at all integer levels

**Location:** `menger-app/src/main/scala/menger/objects/higher_d/TesseractSponge2.scala`,
`menger-app/src/main/scala/menger/objects/higher_d/Mesh4DProjection.scala`,
`optix-jni/src/main/native/shaders/hit_triangle.cu`
**Est. Effort:** 4h
**Reproducer:** `--objects type=tesseract-sponge-2:level=1` or any integer level

**Symptom:** `TesseractSponge2` at integer levels (1, 2, …) renders as a nearly solid dark cube.
Fractional levels (e.g. 1.3) show the expected additional protrusions but both the base cube and
protrusions are dark. The object is geometrically correct (vertex counts, triangle counts, UVs all
pass unit tests) but visually wrong.

**Investigation summary (Sprint 17):**

*Hypothesis 1 — Inverted normals (REJECTED):* Commit 586de54 added centroid-based outward normal
checking in `Mesh4DProjection.quadToTriangleMesh`. The shader at line 90 of `hit_triangle.cu`
already flips the stored normal to face the incoming ray
(`geom.normal = geom.entering ? normal : -normal`), so inverted winding cannot cause darkness.
Reference render taken at d5ef369 (before 586de54) shows the same darkness. Reverted in 48e1eeb.

*Hypothesis 2 — Self-shadowing / shadow occlusion (UNTESTED):* The projected 4D faces may produce
geometry that causes shadow rays to hit the object's own interior faces, resulting in perpetual
shadow. Requires per-face shadow-bias investigation in the shader.

*Hypothesis 3 — Lighting angle (UNTESTED):* The projected faces may happen to be nearly
perpendicular to the default light direction, producing near-zero `dot(N, L)` for most faces.

*Hypothesis 4 — Face normals near zero after projection (UNTESTED):* Degenerate projected faces
(area ≈ 0) get normal `(0, 1, 0)` — may be the majority of faces when viewed from certain angles.

**Known state:** Darkness is pre-existing and unaffected by the Sprint 17 reverts. Manual test 45
("TesseractSponge2 L1.3 fractional") is not passing (dark output), but the fractional geometry
(protrusions) is present — the render is dark but structurally correct. Root cause still unknown.

---

## Medium Priority

## Low Priority

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
