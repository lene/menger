# Code Quality Assessment & Refactoring Guide

**Date:** 2025-11-19
**Branch:** feature/shadow-rays
**Scope:** Complete codebase (13,800 lines Scala + 3,100 lines C++/CUDA)
**Overall Quality Score:** 7.5/10
**Maintainability Score:** 7/10

---

## Executive Summary

The Menger codebase demonstrates **strong architectural discipline** with excellent test coverage (818 passing tests), good separation of concerns, and clean functional programming practices. However, recent additions (shadow rays, multiple lights, camera controls) have revealed opportunities for improvement in:

- **Constant management** - Magic numbers in shaders (1e16f, thresholds)
- **Code duplication** - Shadow/lighting code repeated across shaders
- **Error handling** - Silent failures in native layer
- **API clarity** - FOV parameter semantics unclear

**Before Merge:** All high-priority issues should be addressed to maintain code quality standards.

---

## Part 1: High Priority Issues (Must Fix Before Merge)

### Issue 1.1: Hardcoded Constants in Shadow Ray Tracing

**Category:** Hardcoded Constants (Magic Numbers)
**Severity:** HIGH
**Files:** `optix-jni/src/main/native/shaders/sphere_combined.cu`
**Lines:** 305, 311, 324, 498, 500

**Problem:**
Multiple occurrences of raw constant `1e16f` for shadow ray maximum distance without explanation:

```cuda
// Line 324 (plane miss shader)
tmax: 1e16f,  // tmax (effectively infinite)

// Line 498 (closest hit shader)
tmax: 1e16f,  // tmax (effectively infinite)
```

Also hardcoded `1.0f/255.0f` threshold appears multiple times (lines 305, 311).

**Impact:**
- Unclear that these represent "infinite" distance for shadow rays
- Duplication means changes require multiple edits
- Inconsistent with existing `MAX_RAY_DISTANCE` constant used elsewhere
- Future maintainers may "fix" inconsistencies incorrectly

**Solution:**
Add to `optix-jni/src/main/native/include/OptiXData.h` in RayTracingConstants:

```cpp
namespace RayTracingConstants {
    // Existing constants...

    // Shadow ray constants
    constexpr float SHADOW_RAY_MAX_DISTANCE = 1e16f;  // Effectively infinite for shadow tests
    constexpr float MIN_LIGHTING_THRESHOLD = 1.0f / 255.0f;  // Minimum NdotL to contribute (sub-pixel)
}
```

Then replace all occurrences:

```cuda
// Before
tmax: 1e16f,  // tmax (effectively infinite)
if (ndotl <= 1.0f/255.0f) continue;

// After
tmax: SHADOW_RAY_MAX_DISTANCE,
if (ndotl <= MIN_LIGHTING_THRESHOLD) continue;
```

**Effort:** 30 minutes
**Risk:** Low (pure refactor, no behavior change)

---

### Issue 1.2: Shadow Ray Code Duplication

**Category:** Code Duplication (DRY Violation)
**Severity:** HIGH
**Files:** `optix-jni/src/main/native/shaders/sphere_combined.cu`
**Lines:** 309-342 (plane miss) and 484-519 (sphere closest hit)

**Problem:**
Nearly identical shadow ray tracing code appears in **two places** (34 lines each = 68 lines total):

1. Plane miss shader (lines 309-342)
2. Opaque sphere closest hit shader (lines 484-519)

Both blocks contain identical logic for:
- Shadow origin offset calculation
- optixTrace call with same parameters
- Shadow payload unpacking
- Statistics tracking

**Impact:**
- Bug fixes must be applied twice (maintenance burden)
- Already caused inconsistency (different comments in each location)
- Future shadow features (soft shadows, colored shadows) require duplicate changes
- Increases risk of divergence

**Solution:**
Extract to device function in `sphere_combined.cu`:

```cuda
/**
 * Trace a shadow ray to determine visibility between hit point and light.
 *
 * @param hit_point Surface position where shadow ray originates
 * @param normal Surface normal (for offset to avoid self-intersection)
 * @param light_dir Direction to light source (normalized)
 * @return Shadow factor in range [0, 1] where 0=fully shadowed, 1=fully lit
 */
__device__ float traceShadowRay(
    const float3& hit_point,
    const float3& normal,
    const float3& light_dir
) {
    // Offset origin along normal to avoid shadow acne (self-intersection)
    const float3 shadow_origin = hit_point + normal * SHADOW_RAY_OFFSET;

    // Payload: 0.0 if ray hits (shadowed), 1.0 if ray misses (lit)
    unsigned int shadow_payload = 0;

    optixTrace(
        params.handle,
        shadow_origin,
        light_dir,
        SHADOW_RAY_OFFSET,           // tmin (avoid immediate intersection)
        SHADOW_RAY_MAX_DISTANCE,     // tmax (effectively infinite)
        0.0f,                         // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        1,                            // SBT offset (shadow ray type)
        2,                            // SBT stride (number of ray types)
        1,                            // missSBTIndex (shadow miss)
        shadow_payload
    );

    // Track shadow ray statistics
    if (params.stats) {
        atomicAdd(&params.stats->shadow_rays, 1ULL);
        atomicAdd(&params.stats->total_rays, 1ULL);
    }

    // Convert payload: 0 (hit) → 0.0 (shadowed), 1 (miss) → 1.0 (lit)
    return __uint_as_float(shadow_payload);
}
```

**Usage in both shaders:**

```cuda
// Before (34 lines of duplicated code)
const float3 shadow_origin = hit_point + plane_normal * SHADOW_RAY_OFFSET;
unsigned int shadow_payload = 0;
optixTrace(/* 10 parameters */);
const float shadow_attenuation = __uint_as_float(shadow_payload);
if (params.stats) { /* ... */ }
final_color *= shadow_attenuation;

// After (1 line)
float shadow_factor = traceShadowRay(hit_point, normal, light_dir);
final_color *= shadow_factor;
```

**Files to modify:**
1. `sphere_combined.cu` - Add device function before `__miss__plane()`
2. `sphere_combined.cu` line 309-342 - Replace with function call
3. `sphere_combined.cu` line 484-519 - Replace with function call

**Effort:** 45 minutes
**Risk:** Low (well-tested feature, pure extraction)

---

### Issue 1.3: Light Iteration Code Duplication

**Category:** Code Duplication (DRY Violation)
**Severity:** HIGH
**Files:** `optix-jni/src/main/native/shaders/sphere_combined.cu`
**Lines:** 274-351 (plane miss) and 447-528 (sphere closest hit)

**Problem:**
Near-identical light iteration loops appear in **multiple places** (~80 lines each):

1. Plane miss shader (lines 274-351)
2. Opaque sphere closest hit (lines 447-528)
3. Future: Refraction/reflection paths will need same logic

Each block performs:
- Light type dispatching (directional vs point)
- Distance/attenuation calculation
- Shadow ray tracing
- Light accumulation
- Statistics tracking

**Impact:**
- Adding a new light type (spot, area) requires changes in 3+ places
- Risk of inconsistent behavior (different ndotl clamping, attenuation formulas)
- ~240 lines of near-duplicate code across 3 locations
- Future caustics/global illumination features blocked by this

**Solution:**
Extract comprehensive lighting calculation to device function:

```cuda
/**
 * Calculate total lighting contribution from all lights in scene.
 *
 * Supports multiple light types (directional, point) with shadows,
 * attenuation, and color. Returns combined direct + ambient lighting.
 *
 * @param hit_point Surface position to light
 * @param normal Surface normal (normalized)
 * @return Total lighting color (RGB) including ambient term
 */
__device__ float3 calculateLighting(
    const float3& hit_point,
    const float3& normal
) {
    float3 total_lighting = make_float3(0.0f, 0.0f, 0.0f);

    // Accumulate contribution from each light
    for (int i = 0; i < params.num_lights; ++i) {
        const Light& light = params.lights[i];

        // Calculate light direction and attenuation based on type
        float3 light_dir;
        float attenuation;

        if (light.type == LightType::DIRECTIONAL) {
            // Directional light: parallel rays from infinite distance
            light_dir = normalize(make_float3(
                -light.direction[0],
                -light.direction[1],
                -light.direction[2]
            ));
            attenuation = 1.0f;  // No distance falloff

        } else if (light.type == LightType::POINT) {
            // Point light: radial emission with inverse-square falloff
            const float3 light_pos = make_float3(
                light.position[0],
                light.position[1],
                light.position[2]
            );
            const float3 to_light = light_pos - hit_point;
            const float distance = length(to_light);
            light_dir = to_light / distance;  // Normalize

            // Inverse-square law: I = I₀ / (1 + d²)
            attenuation = 1.0f / (1.0f + distance * distance);
        }

        // Lambertian diffuse: NdotL
        const float ndotl = fmaxf(0.0f, dot(normal, light_dir));

        // Skip lights that don't contribute (sub-pixel threshold)
        if (ndotl <= MIN_LIGHTING_THRESHOLD) continue;

        // Trace shadow ray if shadows enabled
        float shadow_factor = 1.0f;
        if (params.shadows_enabled) {
            shadow_factor = traceShadowRay(hit_point, normal, light_dir);
        }

        // Accumulate light contribution
        const float3 light_color = make_float3(
            light.color[0],
            light.color[1],
            light.color[2]
        );

        total_lighting += light_color * light.intensity * attenuation * ndotl * shadow_factor;
    }

    // Add ambient lighting (prevents pure black shadows)
    const float3 ambient = make_float3(AMBIENT_LIGHT_FACTOR);

    // Combine: ambient + diffuse (energy conserving)
    return ambient + total_lighting * (1.0f - AMBIENT_LIGHT_FACTOR);
}
```

**Usage in all shaders:**

```cuda
// Before (80 lines of light iteration logic)
float3 total_lighting = make_float3(0.0f, 0.0f, 0.0f);
for (int i = 0; i < params.num_lights; ++i) {
    const Light& light = params.lights[i];
    // ... 70 lines of type dispatch, attenuation, shadows, etc.
}
const float3 lit_color = plane_color * total_lighting;

// After (2 lines)
const float3 lighting = calculateLighting(hit_point, normal);
const float3 lit_color = plane_color * lighting;
```

**Files to modify:**
1. `sphere_combined.cu` - Add device function at top of file (before shaders)
2. `sphere_combined.cu` line 274-351 - Replace with function call
3. `sphere_combined.cu` line 447-528 - Replace with function call

**Benefits:**
- Single place to add spot lights, area lights, caustics
- Consistent lighting behavior across all surfaces
- ~160 lines removed
- Future: Easy to add specular, BRDF, etc.

**Effort:** 1.5 hours
**Risk:** Medium (hot path optimization, needs careful testing)

---

### Issue 1.4: Camera FOV Semantic Confusion

**Category:** Clarity of Intent / API Design
**Severity:** HIGH
**Files:**
- `optix-jni/src/main/native/OptiXWrapper.cpp` lines 186-189
- `optix-jni/src/main/scala/menger/OptiXResources.scala` lines 65, 142
- `src/main/scala/menger/input/OptiXCameraController.scala` line 177

**Problem:**
FOV parameter semantics are **inconsistent and confusing**. Code uses **horizontal FOV** (non-standard) but API doesn't make this clear:

```cpp
// OptiXWrapper.cpp line 186-189
// IMPORTANT: fov parameter is HORIZONTAL FOV in degrees
float ulen = std::tan(fov * 0.5f * M_PI / 180.0f);  // Horizontal FOV
float vlen = ulen / aspect_ratio;                    // Vertical derived
```

But Scala code treats it as generic "fov":

```scala
// OptiXResources.scala line 65
val fov = 45f  // Is this horizontal or vertical? UNCLEAR!
renderer.setCamera(eye, lookAt, up, fov)

// OptiXCameraController.scala line 177
optiXResources.updateCamera(eye, lookAt, up)  // No FOV parameter at all!
```

**Industry Standard:** Most graphics APIs (OpenGL, DirectX, Unity, Unreal) use **vertical FOV** as default. This code uses **horizontal FOV** which will confuse developers.

**Impact:**
- Developers unfamiliar with codebase will assume vertical FOV (standard)
- Camera aspect ratio changes produce unexpected FOV changes
- Future maintainers may "fix" this, breaking all existing camera setups
- CLI help text doesn't clarify FOV semantics
- No validation prevents unreasonable values (e.g., 180° FOV)

**Solution:**

**Step 1:** Rename parameter to make semantics explicit

```scala
// OptiXRenderer.scala
def setCamera(
    eye: Array[Float],
    lookAt: Array[Float],
    up: Array[Float],
    horizontalFovDegrees: Float  // RENAMED for clarity
): Unit

def updateCamera(
    eye: Vector3,
    lookAt: Vector3,
    up: Vector3,
    horizontalFovDegrees: Float = 45.0f  // Add parameter with default
): Unit
```

**Step 2:** Add validation

```scala
def setCamera(
    eye: Array[Float],
    lookAt: Array[Float],
    up: Array[Float],
    horizontalFovDegrees: Float
): Unit =
    require(
        horizontalFovDegrees > 0 && horizontalFovDegrees < 180,
        s"Horizontal FOV must be in range (0, 180), got $horizontalFovDegrees"
    )
    setCameraNative(eye, lookAt, up, horizontalFovDegrees)
```

**Step 3:** Update all call sites to use named parameter

```scala
// OptiXResources.scala
private def createCamera(): Unit =
    val eye = Array(cameraPos.x, cameraPos.y, cameraPos.z)
    val lookAt = Array(cameraLookat.x, cameraLookat.y, cameraLookat.z)
    val up = Array(cameraUp.x, cameraUp.y, cameraUp.z)
    renderer.setCamera(eye, lookAt, up, horizontalFovDegrees = 45.0f)  // Named!

def updateCamera(eye: Vector3, lookAt: Vector3, up: Vector3): Unit =
    renderer.setCamera(
        Array(eye.x, eye.y, eye.z),
        Array(lookAt.x, lookAt.y, lookAt.z),
        Array(up.x, up.y, up.z),
        horizontalFovDegrees = 45.0f
    )
```

**Step 4:** Document in scaladoc

```scala
/** Set camera parameters for ray tracing.
  *
  * @param eye Camera position in world space
  * @param lookAt Point camera is aimed at in world space
  * @param up Up vector (typically [0, 1, 0] for Y-up)
  * @param horizontalFovDegrees Horizontal field of view in degrees.
  *                             NOTE: This is HORIZONTAL FOV, which is non-standard.
  *                             Most graphics APIs use vertical FOV. This value is
  *                             aspect-ratio independent and matches OptiX SDK convention.
  *                             Valid range: (0, 180). Typical value: 45.
  */
def setCamera(
    eye: Array[Float],
    lookAt: Array[Float],
    up: Array[Float],
    horizontalFovDegrees: Float
): Unit
```

**Files to modify:**
1. `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala` - Rename parameter, add docs
2. `src/main/scala/menger/OptiXResources.scala` - Update all call sites (3 places)
3. `src/main/scala/menger/input/OptiXCameraController.scala` - Add FOV parameter
4. Any test files that call `setCamera` directly

**Effort:** 1 hour
**Risk:** Medium (API change, requires updating call sites)

---

### Issue 1.5: Missing Error Handling for Light Array Bounds

**Category:** Error Handling
**Severity:** HIGH
**Files:** `optix-jni/src/main/native/OptiXWrapper.cpp` lines 443-455
**Files:** `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala`

**Problem:**
`setLights` validates count but **silently fails** on error:

```cpp
void OptiXWrapper::setLights(const Light* lights, int count) {
    if (count < 0 || count > RayTracingConstants::MAX_LIGHTS) {
        std::cerr << "[OptiX] setLights: count " << count << " out of range [0, "
                  << RayTracingConstants::MAX_LIGHTS << "]" << std::endl;
        return;  // SILENT FAILURE - caller has no idea this failed!
    }
    // ... proceed with copy
}
```

**Impact:**
- Caller has no way to know operation failed
- Invalid light configurations silently ignored (no lights rendered)
- No test coverage for this error path
- JNI binding in Scala has no Try/Option wrapper
- Users will report "lights don't work" with no diagnostic

**Solution:**

**Step 1:** Change C++ to throw exception

```cpp
void OptiXWrapper::setLights(const Light* lights, int count) {
    if (count < 0 || count > RayTracingConstants::MAX_LIGHTS) {
        throw std::invalid_argument(
            "Light count " + std::to_string(count) +
            " out of range [0, " + std::to_string(RayTracingConstants::MAX_LIGHTS) + "]"
        );
    }

    if (lights == nullptr && count > 0) {
        throw std::invalid_argument("Light array is null but count is " + std::to_string(count));
    }

    // ... proceed with copy
}
```

**Step 2:** Update JNI to catch and propagate

```cpp
// JNIBindings.cpp
JNIEXPORT void JNICALL Java_menger_optix_OptiXRenderer_setLightsNative(
    JNIEnv* env, jobject obj, jobjectArray lights_array
) {
    try {
        // ... existing marshaling code ...
        wrapper->setLights(lights, count);

    } catch (const std::invalid_argument& e) {
        // Convert C++ exception to Java exception
        jclass exception_class = env->FindClass("java/lang/IllegalArgumentException");
        env->ThrowNew(exception_class, e.what());

    } catch (const std::exception& e) {
        jclass exception_class = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exception_class, e.what());
    }
}
```

**Step 3:** Wrap in Scala with Try

```scala
// OptiXRenderer.scala
@native private def setLightsNative(lights: Array[Light]): Unit

/** Set multiple light sources for the scene.
  *
  * @param lights Array of light configurations (max 8)
  * @return Success(()) if lights set successfully, Failure with error message if validation fails
  */
def setLights(lights: Array[Light]): Try[Unit] = Try {
    require(
        lights.length > 0 && lights.length <= MAX_LIGHTS,
        s"Light count must be in range [1, $MAX_LIGHTS], got ${lights.length}"
    )
    setLightsNative(lights)
}
```

**Step 4:** Update call sites to handle Try

```scala
// OptiXResources.scala
private def createLights(): Unit =
    lights match
      case Some(lightSpecs) =>
        val lightArray = lightSpecs.map(convertLightSpec).toArray
        renderer.setLights(lightArray) match
          case Success(_) =>
            logger.debug(s"Configured ${lightArray.length} light(s) from CLI specification")
          case Failure(exception) =>
            logger.error(s"Failed to set lights: ${exception.getMessage}")
            throw exception  // Or handle gracefully with default light

      case None =>
        // Default single directional light (backward compatibility)
        // ...
```

**Files to modify:**
1. `optix-jni/src/main/native/OptiXWrapper.cpp` - Throw exceptions on error
2. `optix-jni/src/main/native/JNIBindings.cpp` - Catch and convert to Java exceptions
3. `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala` - Wrap in Try
4. `src/main/scala/menger/OptiXResources.scala` - Handle Try result

**Effort:** 1 hour
**Risk:** Low (improves error handling, existing tests will catch issues)

---

## Part 2: Medium Priority Issues (Should Fix)

### Issue 2.1: MengerCLIOptions Excessive Length

**Category:** Single Responsibility / File Size
**Severity:** MEDIUM
**File:** `src/main/scala/menger/MengerCLIOptions.scala` (326 lines)

**Problem:**
Single file handles too many concerns:
- CLI option definitions (30+ options)
- Custom value converters (4 converters, ~130 lines)
- Validation logic (cross-field validation, ~50 lines)
- Enum definitions (Axis, LightType)
- Case classes (PlaneSpec, LightSpec)

**Solution:**
Split into multiple files:

```
src/main/scala/menger/cli/
  ├── MengerCLIOptions.scala       (option definitions only, ~100 lines)
  ├── CLIValueConverters.scala     (colorConverter, vector3Converter, ~80 lines)
  ├── CLIValidation.scala          (validateOpt logic, ~50 lines)
  └── CLIDomainTypes.scala         (Axis, PlaneSpec, LightSpec, ~60 lines)
```

**Effort:** 1 hour
**Risk:** Low (pure refactor)

---

### Issue 2.2: OptiXWrapper.cpp Render Method Too Large

**Category:** Function Size
**Severity:** MEDIUM
**File:** `optix-jni/src/main/native/OptiXWrapper.cpp` lines 501-622 (122 lines)

**Problem:**
Single method does too much:
- Initialization checks
- Pipeline building
- Buffer allocation/reallocation
- Stats initialization
- Params setup (20+ fields)
- CUDA memory operations
- OptiX launching
- Result copying
- Error handling

**Solution:**
Extract helper methods:

```cpp
void OptiXWrapper::render(int width, int height, unsigned char* output, RayStats* stats) {
    validateInitialized();
    rebuildPipelineIfNeeded();
    ensureBuffersAllocated(width, height);

    Params params = buildRenderParams(width, height);
    uploadParamsToGPU(params);

    launchRender(width, height);
    downloadResults(output, stats, width, height);
}

private:
    void validateInitialized();
    void rebuildPipelineIfNeeded();
    void ensureBuffersAllocated(int width, int height);
    Params buildRenderParams(int width, int height);
    void uploadParamsToGPU(const Params& params);
    void launchRender(int width, int height);
    void downloadResults(unsigned char* output, RayStats* stats, int width, int height);
```

**Effort:** 2 hours
**Risk:** Medium (refactoring hot path)

---

### Issue 2.3: Test Setup Code Duplication

**Category:** DRY Violation (Test Code)
**Severity:** MEDIUM
**Files:** Multiple test files in `optix-jni/src/test/scala/menger/optix/`

**Problem:**
TestScenarios.scala exists but underutilized. Many tests duplicate setup:

```scala
// Repeated ~40 times across test files
renderer.setSphere(0.0f, 0.0f, 0.0f, 0.5f)
renderer.setSphereColor(0.75f, 0.75f, 0.75f, 1.0f)
renderer.setIOR(1.5f)
// ... 10 more lines
```

TestScenarios already provides this setup but tests don't use it consistently.

**Solution:**
Adopt TestScenarios builder pattern everywhere:

```scala
// Before (15 lines)
renderer.setSphere(0.0f, 0.0f, 0.0f, 0.5f)
renderer.setSphereColor(0.75f, 0.75f, 0.75f, 1.0f)
// ... 13 more lines

// After (1 line)
TestScenario.shadowTest(alpha = 1.0f).applyTo(renderer)
```

**Effort:** 2 hours
**Risk:** Low (tests verify behavior)

---

### Issue 2.4: Hardcoded Image Dimensions in Tests

**Category:** Hardcoded Constants
**Severity:** MEDIUM
**Files:** Multiple test files

**Problem:**
Direct use of `(800, 600)` appears ~40 times despite ThresholdConstants providing named constants.

**Solution:**

```scala
import ThresholdConstants.*

// Before
val image = renderer.render(800, 600).get

// After
val image = renderer.render.tupled(STANDARD_IMAGE_SIZE).get
```

**Effort:** 30 minutes
**Risk:** Low

---

### Issue 2.5: Magic Numbers in Input Controllers

**Category:** Hardcoded Constants
**Severity:** MEDIUM
**Files:**
- `src/main/scala/menger/input/CameraController.scala` line 34
- `src/main/scala/menger/input/OptiXCameraController.scala` lines 64-70

**Problem:**
Hardcoded sensitivity values without configurability:

```scala
private val orbitSensitivity = 0.3f      // degrees per pixel
private val panSensitivity = 0.005f      // world units per pixel
private val zoomSensitivity = 0.1f       // distance multiplier
```

**Solution:**
Extract to configuration case class:

```scala
case class CameraControlConfig(
  orbitSensitivity: Float = 0.3f,
  panSensitivity: Float = 0.005f,
  zoomSensitivity: Float = 0.1f,
  minDistance: Float = 0.5f,
  maxDistance: Float = 20.0f,
  minElevation: Float = -89.0f,
  maxElevation: Float = 89.0f
)

class OptiXCameraController(
  // ... existing params ...
  config: CameraControlConfig = CameraControlConfig()
)
```

**Effort:** 45 minutes
**Risk:** Low

---

### Issue 2.6: Var Usage in Input Controllers

**Category:** Functional Programming Adherence
**Severity:** MEDIUM
**File:** `src/main/scala/menger/input/OptiXCameraController.scala` lines 32-62

**Problem:**
10 mutable vars with suppressed warnings. Not thread-safe.

**Solution:**
Use AtomicReference with immutable state:

```scala
case class CameraState(
  eye: Vector3,
  lookAt: Vector3,
  up: Vector3,
  distance: Float,
  azimuth: Float,
  elevation: Float
)

class OptiXCameraController(...):
  private val state = new AtomicReference(CameraState(...))

  private def handleOrbit(deltaX: Int, deltaY: Int): Unit =
    val newState = state.updateAndGet { current =>
      current.copy(
        azimuth = current.azimuth + deltaX * orbitSensitivity,
        // ...
      )
    }
```

**Benefits:**
- Thread-safe
- Testable state transitions
- Can add undo/redo

**Effort:** 1.5 hours
**Risk:** Medium (behavior change, needs testing)

---

## Part 3: New Abstraction Opportunities

### Opportunity 3.1: Light Management Abstraction

**Rationale:** With directional + point lights, adding spot/area lights requires abstraction.

**Current State:**
- Light type dispatch scattered across 3 shader locations
- Attenuation calculation duplicated
- No extensibility

**Proposed:**

```cuda
struct LightSample {
    float3 direction;    // Direction to light (normalized)
    float3 color;        // Light color
    float intensity;     // Brightness multiplier
    float distance;      // Distance to light (INFINITY for directional)
    float attenuation;   // Distance-based falloff
};

__device__ LightSample sampleLight(
    const Light& light,
    const float3& hit_point,
    unsigned int* random_state  // For area lights
) {
    // Single place for all light type logic
}
```

**Benefits:**
- Easy to add spot lights, area lights
- Consistent attenuation
- Ready for Monte Carlo sampling

**Effort:** 2 hours (after Issue 1.3 fixed)

---

### Opportunity 3.2: Ray Type Abstraction

**Current:** Magic indices (0, 1) for ray types

**Proposed:**

```cpp
enum class RayType : unsigned int {
    PRIMARY = 0,
    SHADOW = 1,
    // Future: AMBIENT_OCCLUSION = 2, PHOTON = 3
    COUNT
};

// Usage
optixTrace(
    params.handle, origin, direction, tmin, tmax,
    mask, flags,
    static_cast<unsigned int>(RayType::SHADOW),
    static_cast<unsigned int>(RayType::COUNT),
    static_cast<unsigned int>(RayType::SHADOW),
    payload
);
```

**Benefits:**
- Self-documenting
- Compile-time checks
- Easy to add new ray types

**Effort:** 1 hour

---

### Opportunity 3.3: Shader Payload Abstraction

**Current:** Manual float ↔ uint packing scattered everywhere

**Proposed:**

```cuda
struct ShadowRayPayload {
    float attenuation;

    __device__ static ShadowRayPayload unpack(unsigned int packed) {
        return ShadowRayPayload{__uint_as_float(packed)};
    }

    __device__ unsigned int pack() const {
        return __float_as_uint(attenuation);
    }
};

struct PrimaryRayPayload {
    float3 color;
    unsigned int depth;

    __device__ static PrimaryRayPayload unpack(u32 p0, u32 p1, u32 p2, u32 p3);
    __device__ void pack(u32& p0, u32& p1, u32& p2, u32& p3) const;
};
```

**Benefits:**
- Type-safe
- Clear payload structure
- Easy to change format

**Effort:** 1.5 hours

---

### Opportunity 3.4: Scene Description Format (Future)

**Rationale:** Complex scenes hard to specify via 50+ CLI flags

**Proposed:** YAML scene files

```yaml
scene:
  camera: { position: [0, 0.5, 3.0], lookAt: [0, 0, 0] }
  lights:
    - { type: directional, direction: [-1, 1, -1], intensity: 1.0 }
    - { type: point, position: [2, 3, 1], color: [1.0, 0.8, 0.6] }
  geometry:
    - { type: sphere, center: [0, 0, 0], radius: 0.5 }
```

**Benefits:**
- Shareable scenes
- Version control
- Easier iteration

**Effort:** 4-6 hours (out of scope for this MR)

---

## Part 4: Refactoring Priority & Effort

### Must Fix Before Merge (High Priority)

| Issue | Effort | Risk | Priority |
|-------|--------|------|----------|
| 1.1 Hardcoded Constants | 30 min | Low | 1 |
| 1.2 Shadow Ray Duplication | 45 min | Low | 2 |
| 1.3 Light Iteration Duplication | 1.5 hr | Med | 3 |
| 1.4 Camera FOV Semantics | 1 hr | Med | 4 |
| 1.5 Light Error Handling | 1 hr | Low | 5 |

**Total:** ~4.75 hours

### Should Fix (Medium Priority)

| Issue | Effort | Risk | Priority |
|-------|--------|------|----------|
| 2.3 Test Setup Duplication | 2 hr | Low | 6 |
| 2.4 Hardcoded Test Dimensions | 30 min | Low | 7 |
| 2.5 Input Controller Constants | 45 min | Low | 8 |

**Total:** ~3.25 hours

### Can Defer (Low Priority or Large Effort)

| Issue | Effort | Risk | Notes |
|-------|--------|------|-------|
| 2.1 Split MengerCLIOptions | 1 hr | Low | Nice to have |
| 2.2 Extract Render Helpers | 2 hr | Med | Risky refactor |
| 2.6 Immutable Camera State | 1.5 hr | Med | Thread safety |
| 3.1-3.3 New Abstractions | 4-5 hr | Med | Future work |

---

## Part 5: Refactoring Checklist

### Phase 1: Constants & Duplication (Must Do)

- [ ] **Issue 1.1:** Extract shadow ray constants to OptiXData.h
  - [ ] Add SHADOW_RAY_MAX_DISTANCE constant
  - [ ] Add MIN_LIGHTING_THRESHOLD constant
  - [ ] Replace all occurrences in sphere_combined.cu
  - [ ] Verify compilation

- [ ] **Issue 1.2:** Extract traceShadowRay device function
  - [ ] Create device function in sphere_combined.cu
  - [ ] Replace plane miss shader usage (lines 309-342)
  - [ ] Replace sphere closest hit usage (lines 484-519)
  - [ ] Run ShadowTest suite to verify behavior unchanged

- [ ] **Issue 1.3:** Extract calculateLighting device function
  - [ ] Create device function in sphere_combined.cu
  - [ ] Replace plane miss shader usage (lines 274-351)
  - [ ] Replace sphere closest hit usage (lines 447-528)
  - [ ] Run MultipleLightsTest suite to verify
  - [ ] Run ShadowTest suite to verify

- [ ] **Issue 1.4:** Fix camera FOV semantics
  - [ ] Rename parameter to horizontalFovDegrees in OptiXRenderer
  - [ ] Add validation (0 < fov < 180)
  - [ ] Update OptiXResources call sites (3 places)
  - [ ] Update OptiXCameraController
  - [ ] Add scaladoc explaining horizontal vs vertical FOV
  - [ ] Run CameraTest suite to verify

- [ ] **Issue 1.5:** Add error handling for setLights
  - [ ] Modify OptiXWrapper.cpp to throw exceptions
  - [ ] Update JNIBindings.cpp to catch and convert
  - [ ] Wrap OptiXRenderer.setLights with Try
  - [ ] Update OptiXResources to handle Try result
  - [ ] Add test for invalid light count

### Phase 2: Test Quality (Should Do)

- [ ] **Issue 2.3:** Adopt TestScenarios consistently
  - [ ] Update ShadowTest.scala
  - [ ] Update MultipleLightsTest.scala
  - [ ] Update GeometryTest.scala (if exists)
  - [ ] Verify all tests still pass

- [ ] **Issue 2.4:** Use named dimension constants
  - [ ] Import ThresholdConstants in test files
  - [ ] Replace hardcoded (800, 600) with STANDARD_IMAGE_SIZE
  - [ ] Replace hardcoded (400, 300) with TEST_IMAGE_SIZE

- [ ] **Issue 2.5:** Extract camera config
  - [ ] Create CameraControlConfig case class
  - [ ] Update OptiXCameraController constructor
  - [ ] Update CameraController (if applicable)

### Phase 3: Verification

- [ ] Run full test suite: `sbt test`
  - [ ] All 818 tests passing
  - [ ] No new warnings

- [ ] Run scalafix checks: `sbt "scalafix --check"`
  - [ ] No violations

- [ ] Manual testing:
  - [ ] Launch OptiX window: `sbt "run --optix --sponge-type sphere"`
  - [ ] Test camera controls (orbit, pan, zoom)
  - [ ] Test shadows with multiple lights
  - [ ] Verify visual output unchanged

- [ ] Code review:
  - [ ] No magic numbers in shaders
  - [ ] No duplicate lighting/shadow code
  - [ ] Clear parameter names (horizontalFovDegrees)
  - [ ] Error handling for all native calls

---

## Part 6: Post-Refactoring Metrics

**Before Refactoring:**
- Total lines: 13,800 Scala + 3,100 C++/CUDA = 16,900
- Code duplication: ~400 lines (shadow/lighting loops)
- Magic numbers: 15+ occurrences
- Unclear APIs: 1 (FOV parameter)
- Silent failures: 1 (setLights)

**After Refactoring (Expected):**
- Total lines: ~16,500 (-400 duplication, +60 abstractions, -20 constants)
- Code duplication: <50 lines
- Magic numbers: 0 critical occurrences
- Unclear APIs: 0
- Silent failures: 0

**Quality Score Change:**
- Before: 7.5/10
- After: 8.5/10 (estimated)

---

## Appendix A: Testing Strategy

### Unit Tests
Run after each major change:
```bash
# Test OptiX native layer
sbt "project optixJni" test

# Test full suite
sbt test
```

### Integration Tests
```bash
# Shadow functionality
sbt "testOnly menger.optix.ShadowTest"

# Multiple lights
sbt "testOnly menger.optix.MultipleLightsTest"

# Camera
sbt "testOnly menger.optix.CameraTest"
```

### Visual Verification
```bash
# Basic render
sbt "run --optix --sponge-type sphere --timeout 5"

# Shadows
sbt "run --optix --sponge-type sphere --shadows --timeout 5"

# Multiple lights
sbt "run --optix --sponge-type sphere --shadows \
  --light type=directional direction=-1,1,-1 intensity=0.8 \
  --light type=point position=2,3,1 intensity=0.5"
```

---

## Appendix B: Risk Mitigation

### High Risk Changes
- **Issue 1.3 (Light iteration extraction):** Hot rendering path
  - Mitigation: Comprehensive testing, performance benchmarks
  - Rollback plan: Git revert if performance degrades

### Medium Risk Changes
- **Issue 1.4 (FOV parameter rename):** API change
  - Mitigation: Use named parameters, grep for all usages
  - Rollback plan: Easy to revert, low impact

### Low Risk Changes
- **All constant extractions:** Pure refactoring
  - Mitigation: Compiler catches errors
  - Rollback plan: Trivial to revert

---

## Document Status

**Created:** 2025-11-19
**Purpose:** Guide comprehensive refactoring before merge to main
**Target Branch:** feature/shadow-rays → main
**Expected Completion:** 2025-11-19
**Total Effort:** ~8 hours (Phase 1: 4.75h, Phase 2: 3.25h)

---

**Next Steps:**
1. Review this document with team
2. Execute Phase 1 refactoring (high priority issues)
3. Execute Phase 2 refactoring (medium priority issues)
4. Run full verification checklist
5. Create merge request with before/after metrics
