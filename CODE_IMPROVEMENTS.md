# Code Quality Improvements — Open Issues

**Last Updated:** 2026-04-30

Resolved items are removed from this file entirely — git history is the record of what was fixed.

---

## High Priority


### H-dsl-tesseract-demo-blank — DSL TesseractDemo renders as solid uniform background; glass tesseract completely invisible

**Location:** `menger-app/src/main/scala/examples/dsl/TesseractDemo.scala` (Projection4DSpec),
`menger-app/src/main/scala/menger/engines/scene/MeshFactory.scala` or DSL→spec translation,
`optix-jni/src/main/native/shaders/hit_triangle.cu` (glass path)
**Est. Effort:** 3h
**Reproducer (interactive test 77):** `--scene examples.dsl.TesseractDemo`
(equivalent: `--scene examples.dsl.TesseractDemo -s out.png --headless` — health check confirms
99.91% of pixels = RGB(76, 25, 51), the default background colour)

**Symptom:** The render is a completely uniform maroon background. The render health check reports
99.91% of pixels matching the background colour, meaning the glass tesseract occupies at most 0.09%
of the image (a handful of pixels) or is entirely absent. No glass refraction, reflection, or edge
is visible.

**Expected:** A visible glass tesseract projected from 4D to 3D, showing Fresnel reflections and
refractions of the background, positioned in the centre of the frame.

**Investigation notes (2026-04-27, first observation):**
The DSL scene specifies a custom `Projection4DSpec(eyeW=3.0, screenW=1.5, rotXW=15, rotYW=10,
rotZW=0)` and `Material.Glass`. Three hypotheses:
1. *Custom Projection4DSpec places geometry outside camera frustum:* The default camera is at
   `(0, 2, 5)` looking at origin. If `eyeW=3.0 / screenW=1.5` produces a projection that shifts the
   tesseract too far from the origin, it may fall outside the view. Compare with the CLI equivalent
   `--objects type=tesseract:material=glass` (no custom projection) — if that renders correctly, the
   bug is in the Projection4DSpec wiring.
2. *DSL Projection4DSpec not propagated to the OptiX scene builder:* The `projection` field on the
   Tesseract DSL object may not be translated to the CLI/native projection parameters.
3. *Glass on a uniform-colour background renders transparent:* A glass object in a scene with only a
   single directional light and a uniform maroon background may produce near-zero Fresnel contrast at
   most angles. However, a completely blank 99.91% render is unlikely from pure glass transparency
   alone — some refraction distortion should be visible.
Note: The static suite also includes a TesseractDemo headless test (static test 93); check whether
that test has a passing or blank reference image to determine if this is a recent regression.

---

### H-sponge-showcase-crash — DSL SpongeShowcase crashes: CubeSpongeSceneBuilder rejects mixed sponge types

**Location:** `menger-app/src/main/scala/menger/engines/BaseEngine.scala:74` (`buildSceneFromSpecs`),
`menger-app/src/main/scala/menger/engines/scene/MeshFactory.scala` or scene-builder selection logic
**Est. Effort:** 2h
**Reproducer (interactive test 79):** `--scene examples.dsl.SpongeShowcase`

**Symptom:** The application crashes immediately on startup with:
```
ValidationException: All objects must be cube-sponges for CubeSpongeSceneBuilder.
  Field: 'objectSpecs', Value: 'List(sponge-volume, sponge-surface, cube-sponge)'
```
The SpongeShowcase scene contains three objects: one `sponge-volume`, one `sponge-surface`, and one
`cube-sponge`. The engine selects `CubeSpongeSceneBuilder` despite only one of the three objects
being a cube-sponge, and that builder then correctly rejects the mixed list.

**Expected:** The scene should render all three sponge types simultaneously in one frame, demonstrating
the difference between the three 3D sponge variants (VolumeFilling, SurfaceOnly, CubeSponge).

**Investigation notes (2026-04-27, first observation):**
The builder selection in `buildSceneFromSpecs` (`BaseEngine.scala:74`) appears to dispatch to
`CubeSpongeSceneBuilder` whenever a `cube-sponge` object is present in the spec list, rather than
requiring *all* objects to be cube-sponges. The fix should check whether the selected builder is
compatible with all object types in the scene; if not, fall back to the generic triangle-mesh builder
(which handled mixed scenes in Sprint 18.1 per the TD-5 resolution). Reproduces on every startup —
not intermittent.

---

### H-parametric-film-invisible — ParametricMoebius and ParametricKleinBottleFilm render as blank; no geometry visible

**Location:** `menger-app/src/main/scala/examples/dsl/ParametricMoebius.scala`,
`menger-app/src/main/scala/examples/dsl/ParametricKleinBottleFilm.scala`,
parametric surface tessellation / DSL→mesh pipeline,
`optix-jni/src/main/native/shaders/hit_triangle.cu` (film material path)
**Est. Effort:** 4h
**Reproducers (interactive tests 94 and 96):**
- `--scene examples.dsl.ParametricMoebius --shadows`
- `--scene examples.dsl.ParametricKleinBottleFilm --shadows`

**Symptom:** Both scenes render as a completely uniform maroon background. Render health checks:
- ParametricMoebius: 100.00% of pixels = RGB(76, 25, 51)
- ParametricKleinBottleFilm: 99.39% of pixels ≈ RGB(75, 24, 51)
No parametric surface geometry is visible in either case.

**Expected:** A visible Möbius strip (one-sided, film-material, with shadows) and a Klein bottle
(figure-8 form, film-material, with shadows) each centered in the frame.

**Investigation notes (2026-04-27, first observation):**
Both scenes use parametric surface geometry with `Material.Film`. Three failure modes to investigate:
1. *Parametric geometry not generated / zero triangles:* The tessellation may produce an empty mesh
   silently. Add a triangle-count diagnostic after mesh build to confirm geometry is present.
2. *Film material renders transparent at all angles when no environment is present:* Film material
   relies on Fresnel reflectance; if the specular term is zero at all angles (e.g. due to degenerate
   normals from a non-orientable surface like the Klein bottle), the surface could be fully
   transparent. Check whether removing `Material.Film` and using `Material.Chrome` or a diffuse
   colour makes the surface visible.
3. *Scene geometry placed outside camera frustum:* The camera and object positions may not be set up
   correctly in these DSL scenes, placing the objects behind or to the side of the camera.
The 100% blank Möbius result vs 99.39% nearly-blank Klein bottle is consistent with the Klein bottle
having a small amount of visible self-intersection geometry.
Test both scenes with `Material.Chrome` (which is always visible) to isolate material vs geometry.

---

### H-mixed-frac-int-offscreen — Mixed fractional+integer sponge scene: objects at x=±1.5 appear at frame edges, not in view

**Location:** Default camera position / FOV in `menger-app/src/main/scala/menger/engines/OptiXEngine.scala`
or the camera-setup path used when no explicit `--camera-pos` is given
**Est. Effort:** 1h
**Reproducer (interactive test 48, static aspect):**
```
--objects type=tesseract-sponge:level=1.5:pos=-1.5,0,0 --objects type=tesseract-sponge:level=1:pos=1.5,0,0
```

**Symptom:** The two sponge objects appear at the extreme left and right edges of the rendered frame,
each only partially in view — roughly the inner 15–20% of each object is visible, with the rest
clipped by the frame boundary. The centre of the frame is entirely empty (only the checkered floor is
visible). No useful comparison between the fractional-level and integer-level sponge is possible at
this framing.

**Expected:** Both objects fully visible and roughly symmetric about the centre of the frame,
demonstrating the visual difference between level 1.5 (fractional) and level 1 (integer) TesseractSponge.

**Investigation notes (2026-04-27, first observation):**
The default camera has a fixed position and field-of-view that was calibrated for single-object scenes.
Multi-object scenes at ±1.5 X offset fall near or beyond the horizontal extent of the default frustum.
Other multi-object static tests (e.g., "Mixed 4D sponges" test 55) also use pos=±1.5 and may exhibit
the same issue. The fix is either to widen the default FOV, move the default camera back, or
automatically compute a camera position that fits all objects in the scene.

---

### H-mixed-frac-int-interactive-hang — App hangs permanently when changing 4D viewpoint on mixed fractional+integer sponge scene

**Location:** `menger-app/src/main/scala/menger/engines/OptiXEngine.scala` (4D rotation event handler),
`optix-jni/src/main/native/OptiXWrapper.cpp` (`updateMesh4DProjection`),
`menger-app/src/main/scala/menger/objects/higher_d/TesseractSponge.scala`
**Est. Effort:** 3h
**Reproducer (interactive test 48):**
1. Launch: `--objects type=tesseract-sponge:level=1.5:pos=-1.5,0,0 --objects type=tesseract-sponge:level=1:pos=1.5,0,0` (interactive, no `--headless`)
2. Once the initial render appears, press any 4D rotation key (XW / YW / ZW rotation input)
3. Application becomes unresponsive and does not recover

**Symptom:** The app renders the initial frame (with the objects at frame edges per `H-mixed-frac-int-offscreen`)
but hangs permanently as soon as a 4D viewpoint change is requested. No crash is reported, no error
is logged, and `ESC`/`Ctrl+Q` do not terminate the session.

**Expected:** 4D rotation input should re-project both sponge objects and update the render within
one or two frames, matching the behaviour of single-object TesseractSponge scenes under the same input.

**Investigation notes (2026-04-27, first observation):**
The hang occurs only for multi-object 4D scenes, suggesting the issue is specific to updating two
meshes simultaneously via `updateMesh4DProjection`. Hypotheses:
1. *Second-object update blocks on first:* The engine may update the first sponge's GAS and wait for
   confirmation before updating the second, but the IAS rebuild triggered by the first update races
   with or blocks the second GAS update.
2. *Fractional-level sponge mesh update not implemented:* `updateMesh4DProjection` may not handle
   fractional levels correctly (level=1.5 uses the coverage-blend path, which may require a different
   vertex buffer layout than integer levels). If the update fails silently (cf. `M-project4d-cuda-error-paths`),
   the engine may spin indefinitely.
Reproduce with two integer-level sponges (e.g., level=1 and level=2) as control to isolate whether
the fractional level is load-bearing.

---

## Medium Priority

### M-plane-phong-spec-squared — `miss_plane.cu` Phong specular contribution is `spec²` not `spec`
**Category:** `CORRECTNESS`
**Location:** `optix-jni/src/main/native/shaders/miss_plane.cu:197-199`
**Est. Effort:** 0.25h
Non-metallic plane Phong path:
```cpp
const float spec = powf(...) * plane.specular;
total_color = total_color + make_float3(spec, spec, spec) * spec;
```
The specular value is squared (`spec * spec`) instead of being added linearly. Standard Phong adds `specular_color * spec_intensity`; here `spec` serves as both color and intensity. The highlight is more pinched and dimmer than intended, but visually benign for the default scene (light position produces no on-screen highlight). Pre-existing since Sprint 13.1. Fix: `total_color = total_color + make_float3(spec, spec, spec);`

---

### M-project4d-cuda-error-paths — kernel launch failure is logged but mesh is still registered
**Category:** `POOR_ERROR_HANDLING`
**Location:** `optix-jni/src/main/native/OptiXWrapper.cpp:494-535` (`setTriangleMesh4DQuads`)
**Est. Effort:** 1h
On a `launchProject4DQuadsKernel` error in `setTriangleMesh4DQuads`, the code prints to `std::cerr` but
then continues into AABB readback, `triangle_meshes.push_back`, and returns `mesh_index >= 0`. The
caller treats success as "mesh uploaded"; subsequent IAS/render calls touch undefined vertex memory.
The companion `updateMesh4DProjection` (same file, line 584) gets this right: it returns `-3` on
launch failure. Apply the same pattern in `setTriangleMesh4DQuads`: on `err != cudaSuccess`, free the
just-allocated buffers and return `-1`. The Scala wrapper already requires `meshIdx >= 0` via the
existing JNI return-code convention, so the fix surfaces the failure to callers without API changes.
Same comment for missing `cudaMalloc` / `cudaMemcpy` return-code checks at lines 428–447 — they
silently leak on OOM. Pre-existing in `setTriangleMesh`; new code matched the prior pattern. Worth
fixing both call sites in one pass.

---

## Low Priority

### L-anim4d-recovery-discarded — `result.recover` side-effect discarded
**Category:** `POOR_ERROR_HANDLING`
**Location:** `menger-app/src/main/scala/menger/engines/WithAnimation.scala:145`
**Est. Effort:** 0.25h
`buildAnim4DTrackedOrFallback` does `result.recover { case _ => anim4DState.set(None) }` and then
returns `result` — the `Try` produced by `recover` is discarded. The `anim4DState.set(None)` side
effect still fires (because `recover`'s body is evaluated when the Try is *consumed*, but here it
is materialised then thrown away). Today this happens to work because `recover` materialises the
new Try eagerly, but a reader has to convince themselves that it does. Refactor as
`result.fold(_ => anim4DState.set(None), _ => ())` followed by `result`, or as
`result.transform(s => Success(s), e => { anim4DState.set(None); Failure(e) })`. Same intent,
explicit about the side effect.

---

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

### L-render4d-require-null-message — null branching in the require message itself
**Category:** `NAMING`
**Location:** `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala:423-424`
**Est. Effort:** 0.1h
`require(quads4D != null && quads4D.length % 16 == 0, s"… got ${if quads4D == null then "null" else quads4D.length.toString}")`
combines the null check with the modulus check in the predicate, then re-checks null inside the
message string. Split into two requires: `require(quads4D != null, "quads4D must not be null")`
followed by `require(quads4D.length % 16 == 0, …)`. Half the `// scalafix:ok` annotations needed
and the message becomes unconditional.

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
edges and 2112 triple-shared edges in the level-2 manifold histogram. Activate the
`ignore`d diagnostic in `TopologyDiagnosticSpec` to investigate.

---

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
| H-glass-sponge-skin-diffuse | Sprint 17: `use_coverage_blend` now excludes refractive materials; `use_refractive_coverage_blend` path added (vertex_alpha × Fresnel + (1−α) × continuation); `maxRayDepth` implemented in JNI/shader. Full investigation (glass-sponge-investigation.md) found remaining visible artifacts are physically correct Fresnel reflection of the pink background at grazing angles — not a code bug. Closed. |
| L-cli-monolith: MengerCLIOptions is a 375-line monolith | Scallop registers options during construction; extracting groups into separate `self: ScallopConf =>` traits risks initialization-order issues. File is already organized with clear group separators; accept as-is. |
| L-cli-validation-density: CliValidation repetitive requires-pattern | `isSupplied` must be evaluated lazily inside `validateOpt` lambdas (after argument parsing), not eagerly in a data-driven list. The repetition is load-bearing; accept as-is. The `case Some(_)/None` branches were simplified to `case _` where safe. |
