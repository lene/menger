# Code Quality Improvements — Open Issues

**Last Updated:** 2026-03-24

Resolved items are removed from this file entirely — git history is the record of what was fixed.

---

## High Priority

## Medium Priority

## Low Priority

### L-userguide-missing-example-scene — MixedMetallicShowcase absent from DSL example list
**Location:** `docs/USER_GUIDE.md` section 7.6 "Included Example Scenes"
**Est. Effort:** 0.1h
The DSL example scenes list (section 7.6) does not include `MixedMetallicShowcase`, which was
added in Sprint 13. The integration tests (`test_dsl_scenes`) and the manual test script
(`interactive_tests`) both include it. The USER_GUIDE list is the canonical reference and should
stay in sync.

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

### L-caustics-docstring-order — setShadowPayload docstring misplaced above wrong function
**Location:** `helpers.cu` lines ~20–31
**Est. Effort:** 0.1h
The `/** Set shadow ray payload... */` docblock describing `setShadowPayload` appears immediately
before `accumulateShadowAttenuation`, not above `setShadowPayload` (line 58). The two functions
are in the wrong order relative to their docstrings, so readers see the wrong doc for the first
function they encounter. Swap the functions or move the docstring.

---

### L-caustics-duplicate-config — CausticsConfig and dsl.Caustics duplicate fields and validation
**Location:** `optix-jni/.../RenderConfig.scala` and `menger-app/.../dsl/Caustics.scala`
**Est. Effort:** 0.5h
Both case classes carry identical fields (`photonsPerIteration`, `iterations`, `initialRadius`,
`alpha`) and the same `require` guards. The `Caustics.toCausticsConfig` bridge means any change
to limits must be made in two places. Consider extracting validation constants to a shared object
or letting the DSL type own the constraints and stripping them from `CausticsConfig`.

---

### L-userguide-ior-flag-stale — Sections 6.2 and caustics tutorial use removed --ior flag
**Location:** `docs/USER_GUIDE.md` section 6.2 (lines ~628–631), caustics tutorial (~line 1045)
**Est. Effort:** 0.1h
Section 6.2 "Custom Materials" shows `--ior 1.5` as a standalone CLI flag, and the caustics
tutorial example passes `--ior 1.5` as a top-level option. The `--ior` flag was removed in v0.4.3. These should use `ior=1.5` inside `--objects` syntax instead.

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
| Caustics algorithm limitations | Resolved in Sprint 14 (PPM implemented; remaining limits documented in USER_GUIDE §7.3) |
| L-film-blend: blendFresnelColorsRGBAndSetPayload duplicates scalar body | GPU perf trade-off; acceptable if documented |
| OptiX DSL runtime evaluation | Deferred (Sprint 15) |
