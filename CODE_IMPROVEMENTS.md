# Code Quality Improvements — Open Issues

**Last Updated:** 2026-03-11

Cross-reference with [CODE_REVIEW.md](CODE_REVIEW.md) for resolved items.

---

## Pre-existing Test Failures (non-shadow-related)

### T-triangle-ior — RESOLVED (Sprint 13.2 SBT offset fix)
**Status:** Fixed. The SBT offset fix ensures triangle meshes in single-object mode use the
correct hitgroup records. IOR and colour tests now pass (20/20 TriangleMeshSuite).

---

## High Priority

### H-dead-anyhit — Dead anyhit programs in hit_cylinder.cu
**Location:** `hit_cylinder.cu:298-308` (`__anyhit__cylinder`), `hit_cylinder.cu:321-331`
(`__anyhit__cylinder_shadow`)
**Est. Effort:** 0.5h
Both anyhit programs exist in the source but are never registered in any hitgroup in
`PipelineManager.cpp`. They compile into the PTX but are never called at runtime.
`__anyhit__cylinder_shadow` is a vestige of the reverted anyhit RGB shadow system.
`__anyhit__cylinder` has a stale comment about being "temporarily disabled to diagnose
rotation crash". Both should be removed or properly wired up.

### H-dead-overload — Unused createTriangleHitgroupProgramGroup anyhit overload
**Location:** `OptiXContext.h:79-84`, `OptiXContext.cpp:322-354`
**Est. Effort:** 0.25h
The 3-parameter overload accepting an anyhit module+entrypoint was added for the reverted
colored shadow feature. Never called. Remove.

### H-transparent-shadows-dead — transparent_shadows_enabled is a no-op end-to-end
**Location:** `OptiXData.h:454` (Params field), `OptiXWrapper.cpp:699` (setter),
`RenderConfig.h:34-35`, `JNIBindings.cpp:277-287`, `OptiXRenderer.scala`,
`RenderConfig.scala`, `MengerCLIOptions.scala`
**Est. Effort:** 1h (decision: remove dead plumbing or keep for future re-implementation)
The field is set from Scala through JNI into the Params struct, but no shader reads it.
Currently documented in AD-8 as intentionally retained API surface for future use. If the
feature is not planned for the next sprint, consider removing to avoid dead code.

---

## Medium Priority

### M14 — OptiXEngine exceeds 400-line class size guideline
**Location:** `OptiXEngine.scala` (~430 lines)
**Est. Effort:** 4-6h
**Status:** Partially resolved — extracted `SceneClassifier` and `computeEffectiveMaxInstances`
reduced it from 488 → ~430. Still above the 400-line guideline. Further reduction requires
splitting `createMultiObjectScene`/`rebuildScene` into a separate class. Deferred.

---

### M-shadow-material-inconsistency — Shadow shaders use inconsistent material accessors
**Location:** `hit_triangle.cu:303` (`getInstanceMaterial`), `hit_cylinder.cu:316`
(`getInstanceMaterialPBR`)
**Est. Effort:** 0.25h
Both shadow closesthit shaders only need alpha. Triangle uses the basic 2-parameter
`getInstanceMaterial()`, cylinder uses the 7-parameter `getInstanceMaterialPBR()` fetching
6 unused fields. Should both use the simpler `getInstanceMaterial()` for consistency and
to avoid fetching dead values.

---

### M-legacy-shaders — Legacy standalone shader files duplicate main system
**Location:** `shaders/sphere_combined.cu`, `sphere_raygen.cu`, `sphere_miss.cu`,
`sphere_closesthit.cu`
**Est. Effort:** 2-4h (investigate usage, remove or consolidate)
These files contain an older standalone implementation with their own `Params`, hardcoded
magic numbers, and incompatible data structures (reference `MissData` fields that no longer
exist). They are not part of the main `optix_shaders.cu` include chain. If they serve no
test or demo purpose, they should be removed.

---

### M-emission — RESOLVED (before Sprint 13.2)
`getTriangleMaterial()` now has `out_emission` as a seventh output parameter.  Both Fresnel
blend paths in `hit_triangle.cu` use the emission value.  Closed.

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

### M-naming-constants — Overly literal named constants reduce readability
**Location:** `OptiXData.h` RenderingConstants namespace
**Est. Effort:** 1h
Constants like `COLOR_BLACK = 0.0f`, `FRESNEL_BASE = 1.0f`, `DOT_PRODUCT_ZERO_THRESHOLD = 0.0f`,
`FRESNEL_ONE_MINUS_R0 = 1.0f` add indirection without clarity — they're just restatements of
`0.0f` and `1.0f` with longer names. Constants should encode domain meaning (e.g., `AMBIENT_WEIGHT`)
not literal descriptions of their value. Review and consolidate.

---

### M-color-byte-max-dup — COLOR_BYTE_MAX defined in two namespaces
**Location:** `OptiXData.h` — `RayTracingConstants::COLOR_BYTE_MAX` and
`RenderingConstants::COLOR_BYTE_MAX`
**Est. Effort:** 0.25h
Same constant defined twice. Consolidate to a single definition.

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

### L-stale-new-comment — "// NEW" comment in hit_triangle.cu
**Location:** `hit_triangle.cu:124`
**Est. Effort:** 0.05h
Stale comment from when emission output was added. Remove.

---

### L-dead-structs — Unused structs in OptiXData.h
**Location:** `OptiXData.h` — `Photon` struct (~line 288), `MaterialProperties.normal_texture`
and `roughness_texture` fields
**Est. Effort:** 0.25h
`Photon` is declared but never instantiated. `MaterialProperties` has "(future)" texture fields
that are never used. Remove or mark with a clear "reserved for future" ifdef.

---

### L-dead-constant — PLANE_SOLID_LIGHT_GRAY unused
**Location:** `OptiXData.h:46`
**Est. Effort:** 0.05h
Defined but never referenced. Remove.

---

### L-caustic-scale — Hardcoded caustic_scale = 1.0f
**Location:** `caustics_ppm.cu:866`
**Est. Effort:** 0.25h
Comment says "adjust". Should be a named constant or configurable parameter.

---

### L-cyl-shadow — RESOLVED (Sprint 13.2)
`__closesthit__cylinder_shadow` now correctly encodes alpha using `__float_as_uint(alpha)`,
consistent with the sphere and triangle shadow shaders. The previous no-op implementation
(which effectively meant cylinders cast no shadows) has been replaced.

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
