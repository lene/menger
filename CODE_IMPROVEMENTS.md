# Code Quality Improvements — Open Issues

**Last Updated:** 2026-03-07

Cross-reference with [CODE_REVIEW.md](CODE_REVIEW.md) for resolved items.

---

## Medium Priority

### M14 — OptiXEngine exceeds 400-line class size guideline
**Location:** `OptiXEngine.scala` (~430 lines)
**Est. Effort:** 4-6h
**Status:** Partially resolved — extracted `SceneClassifier` and `computeEffectiveMaxInstances`
reduced it from 488 → ~430. Still above the 400-line guideline. Further reduction requires
splitting `createMultiObjectScene`/`rebuildScene` into a separate class. Deferred.

---

### M-emission — Emission ignored for transparent triangle mesh
**Location:** `hit_triangle.cu` — `getTriangleMaterial()`
**Est. Effort:** 1h
`getTriangleMaterial()` has six output parameters (color, ior, roughness, metallic, specular,
film_thickness) but no `emission`. Both Fresnel blend calls in `hit_triangle.cu` pass
`emission = 0.0f` (hardcoded default). Any semi-transparent triangle mesh with `emission > 0`
will not show its glow in the transparent path. The opaque path and sphere/cylinder shaders are
unaffected.
**Fix:** Add `emission` as a seventh output parameter to `getTriangleMaterial()`.

---

### M-eyew-dup — eyeW scroll formula duplicated in both camera handlers
**Location:** `OptiXCameraHandler.scala:76`, `GdxCameraHandler.scala:64`
**Est. Effort:** 0.5h
Both handlers contain byte-for-byte identical code:
```scala
val eyeW = Math.pow(Const.Input.eyeScrollBase, amountY.toDouble).toFloat + Const.Input.eyeScrollOffset
dispatcher.notifyObservers(RotationProjectionParameters(0, 0, 0, eyeW))
```
Should be extracted to a shared helper (e.g., `CameraHandler` trait method or companion object).
Formula changes require updating both files with no compile-time enforcement.

---

### M-key-dup — factor Map and angle() calculation duplicated between key handlers
**Location:** `GdxKeyHandler.scala`, `OptiXKeyHandler.scala`
**Est. Effort:** 0.5h
Both files contain an identical `factor: Map[Int, Int]` and `angle()` calculation. Should be
extracted to a shared location (companion object or `KeyRotationCalc` helper). Smaller impact
than it was before `KeyPressTracker` eliminated the larger duplication.

---

### M-btn — MouseButton.toGdxButton extension method in wrong file
**Location:** `GdxCameraHandler.scala`
**Est. Effort:** 0.5h
The `toGdxButton` conversion logic belongs in `LibGDXConverters`, not in a handler class.
Pre-existing, not introduced with the handler refactoring.

---

## Low Priority

### L-4d-parser — parseFourDRotationValues in wrong trait
**Location:** `CliValidation.scala`
**Est. Effort:** 0.5h
`parseFourDRotationValues` is domain parsing (string → three floats), not validation. It
belongs in a dedicated parser/converter helper (e.g., `cli/converters/`) to keep `CliValidation`
focused on cross-option rules.

---

### L-scroll-test — Missing Shift+Scroll tests for OptiXCameraHandler
**Location:** `OptiXCameraHandler.scala`
**Est. Effort:** 1h
Sprint 11 plan called for "adjust eyeW on Shift+Scroll" and "clamp eyeW above screenW" tests
that were never written. `OptiXCameraHandler` is in `coverageExcludedPackages` (requires LibGDX
runtime), making unit tests difficult. The boundary case where large negative scroll produces
`eyeW ≈ 1.016f` close to `screenW = 1.0f` is a latent crash risk if `Projection` ever receives
`eyeW ≤ screenW`.

---

### L-doc-scroll — handleScroll return value semantically inconsistent, undocumented
**Location:** `GdxCameraHandler.scala`, `OptiXCameraHandler.scala`
**Est. Effort:** 0.25h
`GdxCameraHandler` returns `false` for Shift+Scroll (lets other handlers act);
`OptiXCameraHandler` returns `true` for all scrolls (consumes). Intentional due to different
multiplexer setups, but undocumented. Add a comment to each handler explaining the choice.

---

### L-doc-esc — ESC return value asymmetry undocumented
**Location:** `GdxKeyHandler.scala`, `OptiXKeyHandler.scala`
**Est. Effort:** 0.25h
`GdxKeyHandler` returns `false` on ESC (does not consume, may let system close window);
`OptiXKeyHandler` returns `true` (consumes, prevents app exit). Correct and intentional, but
undocumented. Both branches should have a brief comment explaining why.

---

### L-doc-sentinel — event.eyeW sentinel pattern undocumented
**Location:** `OptiXEngine.scala:93`
**Est. Effort:** 0.25h
`event.eyeW != Const.defaultEyeW` uses the default value as a sentinel to distinguish
"eyeW-changing event" from "rotation-only event". If a user genuinely wants to reset eyeW to
exactly `defaultEyeW`, the engine would ignore it. The assumption should be documented with a
comment.

---

### L-film-indent — Indentation regression in helpers.cu (cosmetic)
**Location:** `helpers.cu` — `calculateLighting()` comment block
**Est. Effort:** 0.1h
The `// Add ambient lighting` comment block lost its 4-space indentation, leaving it at column 0
while surrounding code is at 4-space indent. Editing artifact from adding thin-film.

---

### L-film-magic — Three unnamed magic numbers in computeThinFilmReflectance
**Location:** `helpers.cu`
**Est. Effort:** 0.5h
- `0.001f` — min cosine clamp
- `1e-8f` — Airy denominator guard
- `106.5f` — CIE Y integral normalisation

All three should be named constants per project standard.

---

### L-film-assert — Vacuous assertion in FilmRenderSuite
**Location:** `FilmRenderSuite.scala`
**Est. Effort:** 0.25h
`filmChannelSpread should be >= 0.0` is trivially true (spread cannot be negative). Replace with
a meaningful assertion (e.g., minimum expected spread when film is active) or convert to
`logger.info` only.

---

### L-cyl-literals — Raw 255.0f/255.99f literals in cylinder fallback path
**Location:** `hit_cylinder.cu`
**Est. Effort:** 0.25h
The diffuse fallback path uses unnamed literals. Named constants `COLOR_SCALE_FACTOR` /
`COLOR_BYTE_MAX` exist elsewhere in the codebase and should be used here.

---

### L-cyl-film — Cylinder thin-film silently skipped without comment
**Location:** `hit_cylinder.cu`
**Est. Effort:** 0.1h
The shader retrieves `film_thickness` via `getInstanceMaterialPBR()` but never uses it (no
thin-film branch exists for the diffuse-only cylinder shader). This known limitation should be
documented with a comment so future developers understand the behaviour.

---

### L-cyl-optb — OPTION B comment implies unresolved design decision
**Location:** `hit_cylinder.cu`
**Est. Effort:** 0.1h
Should either document what OPTION A was and why it was rejected, or drop the "OPTION B" framing.

---

### L-cyl-shadow — Empty __closesthit__cylinder_shadow comment misleading
**Location:** `hit_cylinder.cu`
**Est. Effort:** 0.1h
Says "payload already set for blocked" but the shadow payload is actually set by the any-hit
shader. Comment should clarify the payload convention.

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
| Caustics algorithm limitations | Deferred to future sprint |
| L-film-blend: blendFresnelColorsRGBAndSetPayload duplicates scalar body | GPU perf trade-off; acceptable if documented |
| OptiX DSL runtime evaluation | Deferred (Sprint 15) |
