# Code Quality Improvements — Open Issues

**Last Updated:** 2026-04-19 (Sprint 17 review)

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

### H-glass-sponge-skin-diffuse — Glass sponge skin faces use diffuse shading instead of Fresnel/refraction

**Location:** `optix-jni/src/main/native/shaders/hit_triangle.cu`
**Est. Effort:** 6h
**Reproducer:** `--objects type=sponge-volume:level=1.5:material=glass`

**Symptom:** The skin faces of a fractional glass sponge (those with vertex alpha < 1.0, which
represent the partially-exposed cross-section at the fractional boundary) are shaded as diffuse
surfaces with dark banding, rather than with Fresnel reflection/refraction matching the fully
solid inner faces. Manual test 53 ("3D Fractional glass") shows this as dark bands on the skin
layer.

**Investigation summary (Sprint 17):**

*Attempt 1 — Refractive coverage blend (INTRODUCED BUG, REVERTED):* Commit edb941f added a
`use_refractive_coverage_blend` shader path for faces with `has_vertex_alpha_channel=true`. It
traced 3 independent rays (reflected, refracted, continuation) from each glass skin face and
blended the colors using vertex alpha as the coverage weight. This produced geometrically
incoherent color samples when the independent rays traversed the complex sponge interior
separately, resulting in chaotic pink/magenta distortion many times brighter than expected.
Reverted in 5bd8d38.

*Root cause of attempt 1 failure:* The three independent rays (reflected, refracted, continuation)
do not agree on which part of the sponge interior they are traversing. When blended, colors from
opposite sides of the geometry mix, producing incoherent results. A correct implementation would
need to trace a single ray that accounts for the partial coverage in a physically consistent way
(e.g. Russian roulette between refraction and continuation based on the coverage fraction).

*Current state:* Skin faces use the diffuse coverage blend path (`use_coverage_blend`), which
gives correct geometry but wrong material appearance. The dark banding is the shadow of the sponge
geometry falling on the diffuse skin faces.

**Correct approach (not yet implemented):** Skin faces should use Russian roulette path selection:
with probability = (1 - vertex_alpha), treat as continuation ray (transparent, passing through
the skin boundary); with probability = vertex_alpha, treat as refractive glass. This keeps the
light transport physically consistent without blending three incoherent radiance samples.

*Hypothesis 3 — Insufficient ray depth (UNTESTED):* `maxRayDepth` is not yet implemented in the
OptiX renderer — there is no configurable ray recursion limit. Glass rendering requires multiple
bounces (reflection → refraction → exit). If the implicit hard-coded depth is too shallow, glass
faces may terminate early and return black. Discovered during Sprint 17 task 17.4 (DSL render
settings). **Blocked until `maxRayDepth` is implemented** (planned for a future sprint, see also
`H-glass-sponge-skin-diffuse` scope discussion). Test with both the Russian roulette fix and
an increased ray depth limit to isolate contributions.

---

## Medium Priority

## Low Priority

### L-userguide-missing-example-scene — MixedMetallicShowcase absent from DSL example list
**Location:** `docs/guide/dsl-reference.md` "Included Example Scenes"
**Est. Effort:** 0.1h
The DSL example scenes list (section 7.6) does not include `MixedMetallicShowcase`, which was
added in Sprint 13. The integration tests (`test_dsl_scenes`) and the manual test script
(`interactive_tests`) both include it. The `docs/guide/dsl-reference.md` list is the canonical reference and should
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

### L-area-shape-hardcoded — CliTypes.fromCommonLight hardcodes DISK for area light shape
**Location:** `menger-app/src/main/scala/menger/cli/CliTypes.scala` — `LightSpec.fromCommonLight`
**Est. Effort:** 0.25h
The `Area` case always maps `shape = AreaLightShape.DISK` regardless of the actual
`menger.common.AreaLightShape` value. Since `DISK` is the only shape, this is currently correct,
but when `RECT` or `SPHERE` are added the mapping will silently produce the wrong shape.
Add an exhaustive match from `menger.common.AreaLightShape` → `menger.cli.AreaLightShape` when
new shapes are introduced.

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
**Location:** `docs/guide/user-guide.md` section 6.2 (Custom Materials), `docs/guide/advanced.md` (caustics tutorial)
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
| Caustics algorithm limitations | Resolved in Sprint 14 (PPM implemented; remaining limits documented in `docs/guide/advanced.md` §Caustics) |
| L-film-blend: blendFresnelColorsRGBAndSetPayload duplicates scalar body | GPU perf trade-off; acceptable if documented |
| OptiX DSL runtime evaluation | Deferred (Sprint 15) |
