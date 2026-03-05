# Code Review Findings

**Last Updated:** 2026-03-05
**Purpose:** Track code quality issues identified during comprehensive pre-release reviews

---

## Summary

| Priority | Open | Completed |
|----------|------|-----------|
| High | 0 | 6 |
| Medium | 1 | 33 |
| Low | 4 | 17 |

---

## Open Issues

### Medium Priority

| ID | Description | Location | Est. Hours |
|----|-------------|----------|------------|
| ~~M1~~ | ~~Shader physics code duplication (~200 lines)~~ | ~~hit_sphere.cu / hit_triangle.cu~~ | ~~4-6~~ |
| ~~M2~~ | ~~Zero-vector check missing in normalize()~~ | ~~VectorMath.h:24,74-78~~ | ~~1~~ |
| ~~M3~~ | ~~Missing bounds check in setLights()~~ | ~~SceneParameters.cpp:95-104~~ | ~~0.5~~ |
| ~~M4~~ | ~~Duplicated material preset definitions between DSL and OptiX layers~~ | ~~dsl/Material.scala vs optix/Material.scala~~ | ~~2~~ |

**~~M1~~ - RESOLVED:** Extracted 7 shared physics functions to `helpers.cu`:
- `handleFullyTransparent()` - transparent surface pass-through
- `handleFullyOpaque()` - opaque surface with diffuse shading
- `traceFinalNonRecursiveRay()` - max depth fallback
- `computeFresnelReflectance()` - Schlick approximation
- `traceReflectedRay()` - reflection with stats tracking
- `traceRefractedRay()` - Snell's law refraction
- `applyBeerLambertAbsorption()` - volumetric absorption with optional distance_scale

Reduced total code by ~200 lines while consolidating physics calculations.

**~~M2~~ - RESOLVED (2026-02-18):** Added zero-length guards to both `normalize(float3)` (line 24)
and `normalize3f(float v[3])` (lines 74-78) in `VectorMath.h`. Both now return early (zero vector /
no-op) when length ≤ 0.0f, preventing division by zero.

**~~M3~~ - RESOLVED (2026-02-18):** Added defensive clamp in `SceneParameters::setLights()`:
`if (count > RayTracingConstants::MAX_LIGHTS) count = RayTracingConstants::MAX_LIGHTS;`. Also added
null-pointer guard for the case where `lightsArray == nullptr && count > 0`.

**~~M4~~ - RESOLVED (2026-02-18, completed 2026-03-05):** `dsl/Material.scala` now delegates all
preset values to `OptixMaterial`. Added `Color.fromCommon()` to `dsl/Color.scala` and a private
`fromOptix()` helper in `dsl/Material.scala`. Glass, Water, Diamond, Chrome, Gold, Copper, Film,
Parchment presets and the matte/plastic/metal/glass factory methods all delegate to their
`OptixMaterial` counterparts. `Plastic` and `Matte` now also delegate: named `val Plastic` and
`val Matte` were added to `optix/Material.scala` companion object and `dsl/Material.scala`
delegates via `fromOptix(OptixMaterial.Plastic)` / `fromOptix(OptixMaterial.Matte)`. All material
presets now have a single source of truth.

### High Priority

| ID | Description | Location | Est. Hours |
|----|-------------|----------|------------|
| ~~H6~~ | ~~Duplicated scene classification logic between OptiXEngine and AnimatedOptiXEngine~~ | ~~OptiXEngine.scala:199-230, AnimatedOptiXEngine.scala:131-180~~ | ~~3-4~~ |

**~~H6~~ - RESOLVED (2026-03-05):** Created `menger/engines/SceneClassifier.scala` — a standalone
`object SceneClassifier` in `menger.engines` with `classify`, `isTriangleMeshType`, and
`selectSceneBuilder` methods. The `SceneType` enum was moved from `OptiXEngine.scala` to
`SceneClassifier.scala`. Both `OptiXEngine` and `AnimatedOptiXEngine` now delegate to
`SceneClassifier.*`, removing ~50 lines of duplicated private methods from each engine. The
animated engine also gained `TesseractEdgeSceneBuilder` support (M12) via the shared classifier.
15 unit tests added in `SceneClassifierSuite`.

### Medium Priority (Sprint 12 Findings)

| ID | Description | Location | Est. Hours |
|----|-------------|----------|------------|
| ~~M12~~ | ~~Missing TesseractEdgeSceneBuilder in AnimatedOptiXEngine~~ | ~~AnimatedOptiXEngine.scala:158-165~~ | ~~1~~ |
| ~~M13~~ | ~~No Try wrapping around sceneFunction(t) call~~ | ~~AnimatedOptiXEngine.scala:102~~ | ~~0.5~~ |
| M14 | OptiXEngine exceeds 400-line class size guideline | OptiXEngine.scala (~430 lines) | 4-6 |

**~~M12~~ - RESOLVED (2026-03-05):** Resolved by H6. `AnimatedOptiXEngine` now delegates to
`SceneClassifier.selectSceneBuilder`, which correctly routes tesseract-edge scenes to
`TesseractEdgeSceneBuilder`. The functional gap is closed.

**~~M13~~ - RESOLVED (2026-03-05):** `sceneFunction(t)` in `AnimatedOptiXEngine.render()` is now
wrapped in `Try`. A throwing scene function logs an error with frame/t context and increments the
frame counter (skipping the frame), allowing the animation to continue to completion.

**M14 - PARTIALLY RESOLVED (2026-03-05):** `OptiXEngine.scala` reduced from 488 to ~430 lines
by extracting `SceneClassifier` (H6) and `computeEffectiveMaxInstances` helper. Still above the
400-line guideline. Further reduction would require splitting `createMultiObjectScene`/`rebuildScene`
into a separate class — deferred to a future sprint.

### Low Priority

| ID | Description | Location | Est. Hours |
|----|-------------|----------|------------|
| ~~L1~~ | ~~Configuration DSL~~ | ~~New feature~~ | ~~8-10~~ |
| L2 | Metrics and Telemetry | New feature | 6-8 |
| L3 | Scene graph abstraction | Architecture | 10-12 |
| L4 | Comprehensive benchmarking suite | Tests | 8-10 |
| L5 | Plugin system for geometry types | Architecture | 12-15 |
| ~~L6~~ | ~~Magic number for program group count~~ | ~~PipelineManager.cpp:138~~ | ~~0.5~~ |
| ~~L7~~ | ~~Long function setupShaderBindingTable (116 lines)~~ | ~~PipelineManager.cpp:142-257~~ | ~~2~~ |
| ~~L8~~ | ~~Magic number for max photon threads~~ | ~~CausticsRenderer.cpp:106~~ | ~~0.5~~ |
| ~~L9~~ | ~~Magic number for stack size~~ | ~~OptiXContext.cpp:340~~ | ~~0.5~~ |
| ~~L10~~ | ~~Repeated transparency value 0.02f~~ | ~~MaterialPresets.h:46,57,68~~ | ~~0.5~~ |
| L11 | Exception usage in RAII buffer | CudaBuffer.h:77,89 | 1 |
| ~~L12~~ | ~~Duplicated getFractionalMesh logic in sponge classes~~ | ~~SpongeByVolume.scala:52-74, SpongeBySurface.scala:96-114~~ | ~~1~~ |
| ~~L13~~ | ~~Magic number stride=9 in hit_triangle.cu~~ | ~~hit_triangle.cu:79,211~~ | ~~0.5~~ |
| ~~L14~~ | ~~Option.get usage in Plane.toPlaneColorSpec~~ | ~~dsl/Plane.scala:99,101~~ | ~~0.5~~ |
| ~~L15~~ | ~~Tuple overload explosion in DSL types~~ | ~~dsl/Light.scala, dsl/SceneObject.scala, dsl/Camera.scala~~ | ~~2~~ |
| ~~L16~~ | ~~SceneRegistry populated only on class load — list() unreliable~~ | ~~dsl/SceneRegistry.scala, dsl/SceneLoader.scala~~ | ~~1~~ |

**~~L1~~ - RESOLVED (Sprint 10):** The Scala DSL in `menger-app/src/main/scala/menger/dsl/` fully
supersedes this feature idea. Provides `Scene`, `Camera`, `Light`, `Material`, `Plane`, `Color`,
`Vec3`, `Caustics`, `SceneObject` types with implicit tuple conversions, hex color parsing, material
presets, and CLI integration via `SceneLoader` / `SceneRegistry`. Nine complete example scenes ship
in `examples/dsl/`. Moved to Completed Issues below.

**~~L6~~ - RESOLVED (2026-02-18):** Added `constexpr int NUM_PROGRAM_GROUPS = 12;` with a comment
enumerating the groups (raygen×1 + miss×2 + hitgroups: sphere×2 + tri×2 + cyl×2 + caustics×3) in
`PipelineManager.cpp`. The `createPipeline()` call now uses `NUM_PROGRAM_GROUPS`.

**~~L7~~ - RESOLVED (2026-02-18):** Extracted three private helpers from `setupShaderBindingTable`
in `PipelineManager.cpp`: `createRaygenRecord()`, `createMissRecords()`, and
`createHitgroupRecords()`. The main function now handles cleanup/sizing then delegates to the three
helpers. Declarations added to `PipelineManager.h`.

**~~L8~~ - RESOLVED (2026-02-18):** Added `constexpr int MAX_PHOTON_THREADS_PER_ROW = 1024;` in
`CausticsRenderer.cpp` and replaced the magic literal.

**~~L9~~ - RESOLVED (2026-02-18):** Added
`constexpr unsigned int MIN_CONTINUATION_STACK_SIZE = 49152u;  // 48 KB minimum for metallic cylinders`
in `OptiXContext.cpp` and replaced the magic literal.

**~~L10~~ - RESOLVED (2026-02-18):** Added
`constexpr float DIELECTRIC_ALPHA = 0.02f;  // Near-transparent alpha for glass/water/diamond` near
the top of the `RayTracingConstants` namespace in `MaterialPresets.h`. All three
`mat.color[3] = 0.02f` assignments now use `DIELECTRIC_ALPHA`.

**~~L12~~ - RESOLVED (2026-02-18):** Added `protected def buildFractionalMesh(nextLevelMesh, currentLevelMesh)` to
the `FractionalLevelSponge` trait. `SpongeByVolume.getFractionalMesh` and
`SpongeBySurface.getFractionalMesh` now each reduce to a 4-line call that constructs the two meshes
and delegates to the shared helper.

**~~L13~~ - RESOLVED (2026-02-18):** Added `constexpr unsigned int VERTEX_STRIDE_WITH_ALPHA = 9;`
to `OptiXData.h` (alongside the existing `VERTEX_STRIDE_WITH_UV`). Both `stride >= 9` checks in
`hit_triangle.cu` (lines 79 and 211) now use `VERTEX_STRIDE_WITH_ALPHA`.

**~~L14~~ - RESOLVED (2026-02-18):** `Plane.toPlaneColorSpec` now uses a pattern match on
`(color, checkered)` instead of `color.isDefined` / `color.get` / `checkered.get`, removing the
unsafe `.get` calls.

**~~L15~~ - RESOLVED (2026-02-18):** Removed all redundant Float/Int/Double tuple overloads from
`Directional`, `Point`, `Sphere`, `Cube`, `Sponge`, and `Camera` companion objects. The existing
`given Conversion[(Float,Float,Float), Vec3]` etc. in `Vec3.scala` handle all three numeric types
automatically. Removed now-unused `import scala.annotation.targetName` from `Light.scala` and
`Camera.scala` (retained in `SceneObject.scala` where `@targetName` is still used on multi-arg
overloads).

**~~L16~~ - RESOLVED (2026-02-18):** Created `examples/dsl/SceneIndex.scala` — an `object SceneIndex`
with a `val all: List[Scene]` that eagerly references every example scene, triggering their object
initializers and registry calls. `Main.scala` now evaluates `examples.dsl.SceneIndex` before
calling `SceneLoader.load()`. `SceneLoader`'s `ClassNotFoundException` handler now shows the
populated registry list (or a helpful FQN hint if it's still empty).

---

## Documentation Issues

### ~~DOC-1. Missing API Documentation~~ - RESOLVED

**Resolution:** Added error examples to ObjectSpec scaladoc showing common parse error cases
and their corresponding error messages.

### ~~DOC-2. Outdated TODO Comments~~ - RESOLVED

**Resolution:** After investigation, these TODOs are **not outdated** - they document valid future optimizations:
- Line 287: Spatial hash grid would improve O(n) brute-force photon deposition
- Line 574: Multi-light caustics support (currently only uses first light, unlike direct lighting which loops all lights)

These are legitimate future work items, not stale comments.

### ~~DOC-3. Complex Algorithms Lack Explanation~~ - RESOLVED

**Resolution:** Added inline algorithm comments to:
- `Face.scala` - Menger sponge face subdivision (8 unrotated + 4 rotated sub-faces)
- `SpongeBySurface.scala` - Surface-based generation approach and iteration pattern
- `caustics_ppm.cu` - High-level PPM overview (4 phases, key physics equations)

---

## C++ Code Issues

### ~~CPP-1. Magic Numbers in Shaders~~ - RESOLVED

**Resolution:** Added 7 new named constants to `OptiXData.h` and updated shader files:
- `RAY_PARALLEL_THRESHOLD` (1e-6f) - ray nearly parallel to surface
- `FLUX_EPSILON` (1e-10f) - near-zero flux/area threshold
- `HIT_POINT_RAY_TMIN` (0.0001f) - hit point ray minimum t
- `PHOTON_EMISSION_DISTANCE` (20.0f) - photon start distance behind sphere
- `PHOTON_DISK_RADIUS_MULTIPLIER` (2.0f) - disk radius for sphere coverage
- `SQRT_3` (1.732050808f) - RGB cube diagonal for color distance
- `ALPHA_OPAQUE_BYTE` (255) - fully opaque alpha output

Updated files: `caustics_ppm.cu`, `miss_plane.cu`, `helpers.cu`, `raygen_primary.cu`

### ~~CPP-2. Long Functions in Shaders~~ - RESOLVED

**Resolution:** Extracted helper functions from long shader functions:

**Phase 1 - Fresnel color blending (helpers.cu):**
- `blendFresnelColorsAndSetPayload()` - shared by hit_triangle.cu and hit_sphere.cu
- `payloadToFloat3()` - convert integer payloads to float3

**Phase 2 - Triangle geometry (hit_triangle.cu):**
- `getTriangleGeometry()` - extracts hit point, interpolated normal, UVs
- `getTriangleMaterial()` - gets material properties with texture sampling
- `TriangleGeometry` struct - encapsulates interpolated geometry data

**Phase 3 - Antialiasing (helpers.cu):**
- `sampleGridAndDetectEdges()` - samples 3x3 grid and computes max color difference

**Phase 4 - Photon emission (caustics_ppm.cu):**
- `emitDirectionalPhoton()` - generates parallel rays for directional lights
- `emitPointPhoton()` - generates random rays for point lights
- `calculatePhotonFlux()` - computes per-photon flux from light properties

**Phase 5 - Photon absorption (caustics_ppm.cu):**
- `applyPhotonBeerLambert()` - applies Beer-Lambert absorption to photon flux

**Design decision:** `__intersection__sphere()` (92 lines) left unchanged with documented
rationale - it's adapted from NVIDIA OptiX SDK and refactoring would risk bugs and
make SDK comparison difficult.

---

## Testing Issues

### ~~TEST-1. Insufficient Edge Case Testing~~ - RESOLVED

**Resolution:** Added ~135 edge case tests across 6 test files:
- `ColorConversionsSuite.scala` (NEW) - 19 tests for array bounds, boundary values
- `MaterialUnitSuite.scala` (NEW) - 40 tests for `fromName()`, factory methods, edge values
- `AnimationSpecificationSuite.scala` - +30 tests for frames=0, malformed parsing
- `CubeSuite.scala` - +6 tests for zero/negative scale
- `MaterialConfigSuite.scala` (NEW) - 20 tests for preset validation
- `CameraConfigSuite.scala` (NEW) - 9 tests for default values

**Findings documented:**
- AnimationSpecification parser doesn't support negative ranges (delimiter conflict)
- Material class has no validation for IOR/roughness/metallic values
- Cube handles degenerate cases (scale=0) gracefully

### ~~L6. Property-Based Tests~~ - RESOLVED

**Resolution:** Added ScalaCheck property-based tests for core mathematical types:
- `PropertyTestGenerators.scala` - Generators for Vector[4], Matrix[4], angles, floats
- `VectorPropertySuite.scala` - 18 tests: commutativity, associativity, identity, inverse, triangle inequality
- `MatrixPropertySuite.scala` - 7 tests: identity laws, associativity, vector multiplication
- `RotationPropertySuite.scala` - 10 tests: 360° equivalence, inverse, length preservation, composition

**Key design decisions:**
- Used small value ranges (-10 to 10) for tests involving chained operations to avoid float precision issues
- Added explicit `approxEqual` helper with 0.001 tolerance for property tests while keeping `Const.epsilon` (1e-5) for production code
- Fixed bug in `CustomMatchers.MatricesRoughlyEqualMatcher` (missing `math.abs()`)

---

## Completed Issues (v0.4.1)

### High Priority - All Complete

- ✅ OptiXEngine 22-parameter constructor → single config object
- ✅ Magic numbers extracted to `menger.common.Const`
- ✅ Color conversion deduplication (`ColorConversions.rgbIntsToColor()`)
- ✅ Vector3 conversion deduplication (`Vector3Extensions.toVector3`)
- ✅ Repetitive validation patterns → generic `requires` helper

### Medium Priority - Completed

- ✅ Deep for-comprehension in ObjectSpec simplified (8-level → clean 8-line)
- ✅ Top-level test functions moved to `Face4DTestUtils.scala`
- ✅ Validation error messages improved with actionable guidance
- ✅ Wildcard imports removed from test files
- ✅ Magic numbers in tests extracted to named constants
- ✅ Complex conditionals simplified with pattern matching
- ✅ Method complexity reduced (`setupCubeSponges` → 6 helpers)
- ✅ Boolean expressions simplified (13 expressions, 16 helper methods)
- ✅ Regex documentation and extraction
- ✅ Naming consistency (`timeSpecValid` → `isTimeSpecValid`)
- ✅ Test organization improvements (M8) - renamed files, created test packages
- ✅ Transform matrix utilities (M1) - already in `menger.common.TransformUtil`
- ✅ Extract regex patterns (M4) - moved to `menger.common.Patterns`
- ✅ Specific exception types (M7) - `MengerException` hierarchy in `menger.common`
- ✅ Material helper methods (M2) - `withXxxOpt` methods for cleaner construction
- ✅ Builder pattern review (M5) - Config classes already well-structured
- ✅ Comprehensive error context (M3) - Enhanced error messages across ObjectSpec, CLI converters
- ✅ Strategic debug logging (M10) - Added to ObjectSpec, AnimationSpecification, CliValidation
- ✅ Extract animation parameter parsing (M6) - Constants and helpers in AnimationSpecification
- ✅ Reduce cognitive complexity (M9) - Split CliValidation.registerValidationRules, extracted material helper in OptiXEngine
- ✅ Encapsulate mutable state (M11) - Closed as accepted framework constraint (see Deferred section)

### Medium Priority - Completed (this sprint)

- ✅ M2: Zero-vector guards added to `normalize()` and `normalize3f()` in `VectorMath.h`
- ✅ M3: Defensive bounds clamp and null guard added to `SceneParameters::setLights()`
- ✅ M4: DSL material presets delegate to `OptixMaterial`; single source of truth for all presets including `Plastic`/`Matte`
- ✅ H6: `SceneClassifier` object extracted; both engines delegate to it; `SceneType` enum moved to `SceneClassifier.scala`
- ✅ M12: Resolved by H6 — animated engine now routes tesseract-edge scenes to `TesseractEdgeSceneBuilder`
- ✅ M13: `sceneFunction(t)` wrapped in `Try`; failed frames are logged and skipped gracefully

### Low Priority - Completed

- ✅ Redundant `setSphereColor(r,g,b)` method removed
- ✅ Line length violations fixed (100 char limit)
- ✅ Import organization (scalafix compliant)
- ✅ Redundant type annotations reviewed
- ✅ Test magic numbers extracted
- ✅ Property-based tests for Vector, Matrix, Rotation
- ✅ L1: Configuration DSL - fully implemented as Sprint 10 Scala DSL in `menger-app/src/main/scala/menger/dsl/` with nine example scenes in `examples/dsl/`
- ✅ L6: `NUM_PROGRAM_GROUPS = 12` named constant added to `PipelineManager.cpp`
- ✅ L7: `setupShaderBindingTable` split into `createRaygenRecord`, `createMissRecords`, `createHitgroupRecords`
- ✅ L8: `MAX_PHOTON_THREADS_PER_ROW = 1024` named constant added to `CausticsRenderer.cpp`
- ✅ L9: `MIN_CONTINUATION_STACK_SIZE = 49152u` named constant added to `OptiXContext.cpp`
- ✅ L10: `DIELECTRIC_ALPHA = 0.02f` named constant added to `MaterialPresets.h`; all three usages updated
- ✅ L12: `buildFractionalMesh()` extracted to `FractionalLevelSponge` trait; `SpongeByVolume` and `SpongeBySurface` delegate to it
- ✅ L13: `VERTEX_STRIDE_WITH_ALPHA = 9` added to `OptiXData.h`; both magic `9` checks in `hit_triangle.cu` updated
- ✅ L14: `Plane.toPlaneColorSpec` uses pattern match instead of `Option.get`
- ✅ L15: Redundant Float/Int/Double tuple overloads removed from all DSL companion objects
- ✅ L16: `SceneIndex` object forces eager initialization of all example scenes; `Main.scala` evaluates it before scene loading

---

## Deferred / Accepted Issues

The following issues were explicitly deferred or accepted:

1. **Mutable state in LibGDX integration** - Required by framework, properly suppressed
2. **M11: Encapsulate mutable state in input controllers** - After analysis, encapsulating into case classes would add complexity without benefit. The existing code is already well-structured: vars have proper suppression annotations, state is properly encapsulated (private with controlled access), and the SphericalOrbit trait provides clean abstraction. LibGDX's InputAdapter pattern requires mutable state for tracking between callbacks.
3. **OptiX cache management** - Works correctly, no changes needed
4. **Caustics algorithm issues** - Deferred to future sprint, not blocking
5. **Test performance** - Acceptable, optimization not priority
6. **Exception-based error handling** - Some `throw` usage required at boundaries
7. **L11: Exception usage in CudaBuffer (CudaBuffer.h:77,89)** - Accepted. Throwing at JNI boundaries is the correct pattern: Java callers cannot receive CUDA error codes any other way. The constructor throws `CudaException` on allocation failure, which propagates cleanly to Scala as a JVM exception. No change needed.

---

## Estimated Remaining Effort

| Category | Hours |
|----------|-------|
| ~~High priority (H6)~~ | ~~3-4~~ ✅ |
| Medium priority (M14 only) | 4-6 |
| Low priority (feature ideas L2–L5) | ~36 |
| Documentation | 0 |
| C++ Issues | 0 |
| **Total** | **~40-42 hours** |

**Note:** H6, M12, M13 resolved in Sprint 12 code review fixes. M14 (`OptiXEngine` ~430 lines,
above 400-line guideline) deferred — would require splitting `createMultiObjectScene`/`rebuildScene`.

---

## Review Process

Before each release:

1. Run `sbt "scalafix --check"` - verify lint compliance
2. Run `sbt compile` - ensure no warnings
3. Review this document for open issues
4. Perform fresh code review of changed files
5. Update this document with new findings
6. Mark completed items

---

## Notes

**Good Practices Already in Place:**
- No `var` or `throw` in production code (wartremover enforced)
- Functional style throughout
- Comprehensive test coverage (~1617 tests, ~85% statement coverage)
- Coverage protection with ratchet mechanism (60% floor, 80% min, 1% drop threshold)
- Scalafix integration for consistent style
- DSL types use `require()` for all validated fields (ior, roughness, metallic, specular, emission, size) — runtime validation catches misuse early
- `TriangleMeshData` cleanly separated in `menger-common` with `menger-app` re-exporting via a one-line `export` (no real duplication)
- `helpers.cu` `traceContinuationRay` / `COVERAGE_CONTINUATION_OFFSET` and `FractionalLevelSponge.SkinNormalOffset` are well-documented with the coupling between Scala and CUDA constants explained inline
- DSL `Color` provides an implicit `Conversion[String, Color]` for ergonomic hex literal syntax in scene definitions
- `SceneLoader` error handling uses `Either[String, LoadedScene]` throughout and distinguishes `ClassNotFoundException` / `NoSuchFieldException` / general exceptions with distinct messages
- **Sprint 12 positives:**
  - `LoadedScene` ADT cleanly separates `Static` vs `Animated` scenes with pattern matching
  - `SceneConverter` extracted as a reusable utility, eliminating duplication between `Main` and `AnimatedOptiXEngine`
  - `TAnimationConfig` is a pure value type with deterministic `tForFrame()` interpolation
  - `SceneLoader` reflection-based detection of `def scene(t: Float)` is well-encapsulated with proper error handling
  - CLI validation for t-parameter options is thorough (27 new tests covering mutual exclusivity, requirement chains)
  - Example animated scenes (`OrbitingSphere`, `PulsingSponge`) are clean and demonstrate the pattern well
