# Code Quality Improvements — Open Issues

**Last Updated:** 2026-03-16

Cross-reference with [CODE_REVIEW.md](CODE_REVIEW.md) for resolved items.

---

## Pre-existing Test Failures (non-shadow-related)

### T-triangle-ior — RESOLVED (Sprint 13.2 SBT offset fix)
**Status:** Fixed. The SBT offset fix ensures triangle meshes in single-object mode use the
correct hitgroup records. IOR and colour tests now pass (20/20 TriangleMeshSuite).

---

## High Priority

### H-dead-anyhit — RESOLVED (Sprint 13.2 dead code cleanup)
**Status:** Fixed. Both `__anyhit__cylinder` and `__anyhit__cylinder_shadow` have been removed.
They were never registered in any hitgroup in PipelineManager.cpp and compiled into PTX as
unreachable dead code.

### H-dead-overload — RESOLVED (Sprint 13.2 dead code cleanup)
**Status:** Fixed. The 4-parameter `createTriangleHitgroupProgramGroup` (CH+AH) overload and
the 6-parameter `createHitgroupProgramGroup` (CH+AH+IS) overload have been removed from
OptiXContext.cpp and OptiXContext.h. Neither was ever called.

### H-transparent-shadows-dead — RESOLVED (Sprint 13.2 colored shadows)
**Status:** Fixed. The `transparent_shadows_enabled` field in Params is now actively read by
shaders (`closesthit__shadow`, `closesthit__triangle_shadow`, `closesthit__cylinder_shadow`)
to gate colored shadow attenuation via Beer-Lambert absorption. No longer dead code.

---

## Medium Priority

### M-userguide-version-header — USER_GUIDE.md header fields are stale
**Location:** `docs/USER_GUIDE.md` lines 3–4
**Est. Effort:** 0.1h
The guide header reads `**Version**: 0.5.2` and `**Last Updated**: February 2026`. The file was
modified in Sprint 13 (March 2026) to add colored shadows, plane materials, and the material
reference section. The "Last Updated" field should be updated to March 2026 whenever the guide
changes; leaving it stale misleads readers about whether documentation is current.

---

### M-userguide-t-animation-version — t-parameter animation labelled as "v0.6.0"
**Location:** `docs/USER_GUIDE.md` section 7.2 (line ~911 in current file)
**Est. Effort:** 0.1h
Section 7.2 opens with "**Introduced in v0.6.0**". The t-parameter animation system was in fact
introduced in v0.5.2 (see CHANGELOG.md `[0.5.2]` → Added). The current version string in
`MengerCLIOptions.scala` is also `0.5.2`. This is an inaccurate version tag that will confuse
users trying to determine whether a feature is available in their installation.

---

### M-userguide-deprecated-flags — Tutorial 8.2 uses removed CLI flags
**Location:** `docs/USER_GUIDE.md` section 8.2 "Creating Glass Objects"
**Est. Effort:** 0.25h
The "Good Glass Setup" example in Tutorial 8.2 passes `--ior 1.5` and `--radius 1.5` as
top-level options. Both flags were removed in v0.4.3 (CHANGELOG: "Removed: `--radius`, `--ior`,
`--scale`, `--center`"). A user following this tutorial will get a parse error. The example
should be rewritten using `--objects 'type=sphere:ior=1.5:size=1.5'` per the modern syntax.

---

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

### L-userguide-missing-example-scene — MixedMetallicShowcase absent from DSL example list
**Location:** `docs/USER_GUIDE.md` section 7.6 "Included Example Scenes"
**Est. Effort:** 0.1h
The DSL example scenes list (section 7.6) does not include `MixedMetallicShowcase`, which was
added in Sprint 13. The integration tests (`test_dsl_scenes`) and the manual test script
(`interactive_tests`) both include it. The USER_GUIDE list is the canonical reference and should
stay in sync.

---

### L-roadmap-stale-date — ROADMAP.md "Last Updated" not refreshed in Sprint 13
**Location:** `ROADMAP.md` line 3
**Est. Effort:** 0.05h
ROADMAP.md carries `**Last Updated:** 2026-03-07` but was modified during Sprint 13 (2026-03-12)
to mark Sprint 13 complete and add Sprint 14 task 14.8. The date should be updated when the
file changes.

---

### L-arc42-test-count-stale — arc42 section 11.4 monitoring threshold is stale
**Location:** `docs/arc42/11-risks-and-technical-debt.md` section 11.4 Monitoring
**Est. Effort:** 0.1h
The monitoring table reads "Test count | CI | < 1091 (regression)". The project now has 1629+
tests (SPRINT13.md success criteria, CHANGELOG v0.5.2). The regression threshold should be
updated to reflect the current baseline; otherwise CI would not alert until the count fell below
an already-obsolete floor.

---

### L-changelog-duplicate-version — CHANGELOG.md has duplicate [0.4.2] header
**Location:** `CHANGELOG.md` lines ~212 and ~260
**Est. Effort:** 0.1h
The `[0.4.2]` version section header appears twice. The second entry (line ~260) covers
Tesseract and cylinder edge rendering and should be `[0.4.1]` or `[0.4.2-preview]`. As written,
the duplicate makes it ambiguous which changes belong to which release, and the diff link at the
bottom for `[0.4.2]` can only point to one commit range. Pre-existing.

---

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

---

### L-cli-monolith — MengerCLIOptions is a 375-line monolith
**Location:** `menger-app/src/main/scala/menger/MengerCLIOptions.scala`
**Est. Effort:** 3h
All 40+ option definitions live inline. Option groups and their defaults could be extracted into
separate option-group traits/objects; validation rules already live in `CliValidation` but more
could move there. Reduces cognitive load when adding new options.

---

### L-cli-validation-density — CliValidation.scala is dense with repetitive validation patterns
**Location:** `menger-app/src/main/scala/menger/cli/CliValidation.scala` (313 lines)
**Est. Effort:** 2h
15+ `validateOpt()` / `requiresOptix()` calls follow the same pattern. A data-driven validation
builder (e.g., a list of `(option, condition, message)` tuples) would halve the line count and
make adding new validations trivial.

---

### L-scene-builder-registry — Builder selection is hardcoded in OptiXEngine
**Location:** `menger-app/src/main/scala/menger/engines/OptiXEngine.scala`
**Est. Effort:** 1h
`selectMeshBuilder()` is a match expression that maps object type strings to builder instances.
A `Map[String, SceneBuilder]` registry would be easier to extend and test independently.

---

### L-converter-duplication — Coordinate parsing repeated across converters
**Location:** `menger-app/src/main/scala/menger/cli/converters/`
**Est. Effort:** 1h
`vector3Converter`, `planeSpecConverter`, and parts of `lightSpecConverter` all parse
comma-separated floats with similar error-handling boilerplate. A shared
`parseFloatComponents(input, expectedCount)` helper in `ConverterUtils` would reduce duplication.

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
