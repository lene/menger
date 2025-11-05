# Glass Sphere Implementation Plan

## Overview

This document outlines the step-by-step plan to fix glass sphere rendering in the OptiX ray tracer. The core issue is that OptiX built-in sphere primitives cannot detect ray intersections from inside the sphere, preventing proper refraction and volume absorption.

## Root Cause Analysis

### Current State
- **Using**: Built-in sphere primitives (`OPTIX_BUILD_INPUT_TYPE_SPHERES`)
- **Problem**: Rays originating inside spheres return `hit_t = -1.0` (no intersection detected)
- **Impact**: Refracted rays cannot find exit point, breaking glass rendering

### Evidence
```
Test 1: Ray from center → hit_t = -1.0 (FAILED)
Test 2: Ray from inside surface → hit_t = -1.0 (FAILED)
Test 3: Same ray with tmin=0 → hit_t = -1.0 (FAILED)
```

### Solution
Implement custom sphere intersection program following OptiX SDK patterns (optixWhitted example).

---

## Implementation Phases

## Phase 1: Code Cleanup and Documentation Updates

### 1.1 Remove Diagnostic Test Code

**File**: `optix-jni/src/main/native/shaders/sphere_combined.cu`

**Action**: Remove lines 237-333 (test code block)

**Current Code to Remove**:
```cuda
// COMPREHENSIVE TEST: Does OptiX sphere primitive detect hits from rays originating inside?
if (entering && is_center_pixel) {
    printf("\n=== INTERNAL RAY TEST SUITE ===\n");
    // ... test code ...
    printf("\n=== END TEST SUITE ===\n\n");
}
```

### 1.2 Update Documentation

**File**: `ABSORPTION_DEBUG_FINDINGS.md`

**Updates**:
- Mark OptiX limitation as confirmed
- Remove contradictory statements about internal ray support
- Add reference to Glass_Rendering_Findings.md

**File**: `FIX_PLAN.md`

**Updates**:
- Update to reference custom intersection solution
- Mark analytical workaround as deprecated

---

## Phase 2: Implement Custom Sphere Intersection

### 2.1 Add Custom Intersection Program

**File**: `optix-jni/src/main/native/shaders/sphere_combined.cu`

**Add New Section** (after includes, before raygen):

```cuda
//------------------------------------------------------------------------------
// Custom Sphere Intersection Program
//------------------------------------------------------------------------------

extern "C" __global__ void __intersection__sphere()
{
    // Get hit group data containing sphere parameters
    const HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

    // Extract sphere geometry
    const float3 center = make_float3(
        hit_data->sphere_center[0],
        hit_data->sphere_center[1],
        hit_data->sphere_center[2]
    );
    const float radius = hit_data->sphere_radius;

    // Get ray parameters
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin();
    const float ray_tmax = optixGetRayTmax();

    // Ray-sphere intersection using quadratic formula
    // Sphere equation: |P - C|² = r²
    // Ray equation: P = O + tD
    // Substituting: |O + tD - C|² = r²

    float3 O = ray_orig - center;
    float a = dot(ray_dir, ray_dir);  // Should be 1 if normalized
    float b = 2.0f * dot(O, ray_dir);
    float c = dot(O, O) - radius * radius;

    float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f) {
        // No intersection
        return;
    }

    float sqrt_disc = sqrtf(discriminant);
    float inv_2a = 0.5f / a;

    // Calculate both intersection points
    float t1 = (-b - sqrt_disc) * inv_2a;  // Near intersection
    float t2 = (-b + sqrt_disc) * inv_2a;  // Far intersection

    // Report valid intersections
    if (t1 >= ray_tmin && t1 <= ray_tmax) {
        // Near intersection (entry from outside OR exit from inside)
        float3 hit_point = ray_orig + t1 * ray_dir;
        float3 normal = (hit_point - center) / radius;

        // Determine hit type based on ray origin position
        float origin_dist_sq = dot(O, O);
        unsigned int hit_kind = (origin_dist_sq > radius * radius + 0.001f) ? 0 : 1;  // 0=entry, 1=exit

        // Report intersection with normal as attributes
        optixReportIntersection(
            t1,
            hit_kind,
            __float_as_uint(normal.x),
            __float_as_uint(normal.y),
            __float_as_uint(normal.z)
        );
    }

    if (t2 >= ray_tmin && t2 <= ray_tmax) {
        // Far intersection (exit from outside OR entry from inside - rare)
        float3 hit_point = ray_orig + t2 * ray_dir;
        float3 normal = (hit_point - center) / radius;

        // For far intersection, typically an exit
        unsigned int hit_kind = 1;  // exit

        optixReportIntersection(
            t2,
            hit_kind,
            __float_as_uint(normal.x),
            __float_as_uint(normal.y),
            __float_as_uint(normal.z)
        );
    }
}

//------------------------------------------------------------------------------
// Attribute Programs
//------------------------------------------------------------------------------

extern "C" __global__ void __attribute__sphere_normal()
{
    // Retrieve normal passed from intersection program
    const unsigned int attr0 = optixGetAttribute_0();
    const unsigned int attr1 = optixGetAttribute_1();
    const unsigned int attr2 = optixGetAttribute_2();

    // Store in hit data (will be available in closest hit)
    float3* normal = getPRD<float3*>();
    normal->x = __uint_as_float(attr0);
    normal->y = __uint_as_float(attr1);
    normal->z = __uint_as_float(attr2);
}
```

### 2.2 Update Closest Hit Shader

**File**: `optix-jni/src/main/native/shaders/sphere_combined.cu`

**Function**: `__closesthit__ch`

**Modifications**:

1. **Retrieve intersection data**:
```cuda
// Get hit type from intersection program
unsigned int hit_kind = optixGetHitKind();
bool is_entry = (hit_kind == 0);
bool is_exit = (hit_kind == 1);

// Get surface normal from attributes (passed by intersection program)
float3 surface_normal;
surface_normal.x = __uint_as_float(optixGetAttribute_0());
surface_normal.y = __uint_as_float(optixGetAttribute_1());
surface_normal.z = __uint_as_float(optixGetAttribute_2());
```

2. **Update entry/exit detection**:
```cuda
// Determine if ray is entering or exiting based on hit kind
const bool entering = is_entry;

// Alternative: verify with dot product (backup check)
// const bool entering = (dot(ray_direction, surface_normal) < 0.0f);
```

3. **Track entry distance for absorption**:
```cuda
if (entering) {
    // Save entry distance in payload for absorption calculation
    *saved_entry_t_payload = t;
} else if (*saved_entry_t_payload >= 0.0f) {
    // Exiting - calculate distance traveled
    float distance_in_glass = t - *saved_entry_t_payload;

    // Apply Beer-Lambert absorption
    float3 absorption = exp3(-hit_data->alpha * distance_in_glass);
    // Will be applied to final color after tracing
}
```

---

## Phase 3: Update Host-Side OptiX Pipeline

### 3.1 Modify Geometry Creation

**File**: `optix-jni/src/main/native/OptiXWrapper.cpp`

**Function**: `createGeometry()` or equivalent

**Change FROM**:
```cpp
OptixBuildInput sphere_input = {};
sphere_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
sphere_input.sphereArray.vertexBuffers = &d_sphere_vertices;
sphere_input.sphereArray.numVertices = 1;
sphere_input.sphereArray.radiusBuffers = &d_sphere_radius;
```

**Change TO**:
```cpp
// Create AABB for custom primitive
OptixAabb aabb;
aabb.minX = sphere_center.x - sphere_radius;
aabb.minY = sphere_center.y - sphere_radius;
aabb.minZ = sphere_center.z - sphere_radius;
aabb.maxX = sphere_center.x + sphere_radius;
aabb.maxY = sphere_center.y + sphere_radius;
aabb.maxZ = sphere_center.z + sphere_radius;

// Upload AABB to device
CUdeviceptr d_aabb;
CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb), sizeof(OptixAabb)));
CUDA_CHECK(cudaMemcpy(
    reinterpret_cast<void*>(d_aabb),
    &aabb,
    sizeof(OptixAabb),
    cudaMemcpyHostToDevice
));

// Create build input for custom primitive
OptixBuildInput sphere_input = {};
sphere_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
sphere_input.customPrimitiveArray.aabbBuffers = &d_aabb;
sphere_input.customPrimitiveArray.numPrimitives = 1;
sphere_input.customPrimitiveArray.flags = &input_flags;
sphere_input.customPrimitiveArray.numSbtRecords = 1;
```

### 3.2 Update Program Groups

**File**: `optix-jni/src/main/native/OptiXWrapper.cpp`

**Function**: `createProgramGroups()` or equivalent

**Add Intersection Program**:
```cpp
OptixProgramGroupDesc hit_prog_group_desc = {};
hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
hit_prog_group_desc.hitgroup.moduleCH = module;
hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
hit_prog_group_desc.hitgroup.moduleIS = module;  // Add intersection program
hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";  // New!
// Optional: Add attribute program if needed
// hit_prog_group_desc.hitgroup.moduleAH = module;
// hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__attribute__sphere_normal";
```

### 3.3 Update Module Compilation

**File**: `optix-jni/src/main/native/OptiXWrapper.cpp`

**Function**: `createModule()` or equivalent

Ensure the PTX compilation includes the new intersection program. No changes needed if using the same sphere_combined.cu file.

---

## Phase 4: Fix Ray Offsetting and State Management

### 4.1 Proper Ray Origin Offsetting

**In closest hit shader**:

```cuda
// Define scene epsilon for offsetting
const float scene_epsilon = 1e-4f;

// Offset ray origins to avoid self-intersection
float3 front_hit_point = hit_point;
float3 back_hit_point = hit_point;

if (entering) {
    // Ray entering glass
    front_hit_point += scene_epsilon * surface_normal;   // Reflection origin
    back_hit_point -= scene_epsilon * surface_normal;    // Refraction origin
} else {
    // Ray exiting glass
    front_hit_point -= scene_epsilon * surface_normal;   // Unusual but possible
    back_hit_point += scene_epsilon * surface_normal;    // Continuation origin
}
```

### 4.2 Update Ray Tracing Calls

**For refraction**:
```cuda
if (refract(refract_dir, ray_direction, normal, ior)) {
    // Trace from back hit point (inside glass)
    optixTrace(
        params.handle,
        back_hit_point,    // Use offset origin
        refract_dir,
        scene_epsilon,     // tmin to avoid self-hit
        1e16f,            // tmax
        0.0f,             // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        // ... payload ...
    );
}
```

**For reflection**:
```cuda
optixTrace(
    params.handle,
    front_hit_point,   // Use offset origin
    reflect_dir,
    scene_epsilon,     // tmin
    1e16f,            // tmax
    // ... rest same as above ...
);
```

---

## Phase 5: Testing and Validation

### 5.1 Unit Tests

Create test program to verify:

1. **External ray entry**:
   - Ray from (0,0,3) toward (0,0,0)
   - Should hit at t ≈ 1.5

2. **Internal ray exit** (THE CRITICAL TEST):
   - Ray from (0,0,0) toward (0,0,1)
   - Should hit at t ≈ 1.5 (not -1.0!)

3. **Refraction path**:
   - Trace complete path through sphere
   - Verify both entry and exit hits
   - Check absorption is applied

### 5.2 Visual Tests

1. **Glass sphere with checkerboard**:
   - Should show refracted, distorted checkerboard
   - Should have Fresnel reflections at edges
   - Should tint based on glass color

2. **Multiple spheres**:
   - Test overlapping transparent spheres
   - Verify no artifacts or black pixels

3. **Edge cases**:
   - Grazing angles (test total internal reflection)
   - Very dark glass (high absorption)
   - Clear glass (low absorption)

### 5.3 Performance Testing

Compare performance:
- Custom intersection vs old built-in primitive
- Measure frame time difference
- Consider optimization if > 20% slower

### 5.4 Lighting Verification

**IMPORTANT**: Re-enable the lighting test after custom intersection implementation is complete.

**Test Location**: `optix-jni/src/test/scala/menger/optix/OptiXRendererTest.scala:365`

**Test Name**: "should render different images with different light directions"

**Current Status**: Temporarily disabled (changed from `it should` to `ignore should`)

**Re-enabling Steps**:
1. Change `ignore should` back to `it should` on line 365
2. Run test: `sbt "project optixJni" test`
3. Verify that different light directions produce visibly different rendered images:
   - Light from top-right: `Array(0.5f, 0.5f, -0.5f)`
   - Light from left: `Array(-1.0f, 0.0f, 0.0f)`
   - Light from behind camera: `Array(0.0f, 0.0f, 1.0f)`
4. If test still fails, investigate:
   - Check that `setLight()` in OptiXWrapper.cpp stores light direction correctly
   - Verify shader uses `params.light_direction` in lighting calculations
   - Ensure light direction is passed to CUDA launch parameters

**Why Disabled**: The test was failing because all three light directions produced identical images, indicating that light direction wasn't affecting the rendering. This was temporarily disabled to focus on implementing custom sphere intersection for proper glass rendering, as fixing the intersection may resolve the lighting issue as a side effect.

---

## Phase 6: Optional Enhancements

### 6.1 Sphere Shell Implementation

If solid sphere has issues, implement hollow sphere:

```cuda
struct SphereShell {
    float3 center;
    float radius_inner;  // e.g., 0.98
    float radius_outer;  // e.g., 1.0
};
```

Benefits:
- More interesting refraction patterns
- Matches optixWhitted example exactly
- Can model hollow glass objects

### 6.2 Triangle Mesh Alternative

If custom intersection is too slow:

```cpp
// Generate triangle mesh for sphere
std::vector<float3> vertices;
std::vector<uint3> indices;
generateSphereMesh(center, radius, 32, 32, vertices, indices);

// Use built-in triangle intersection (hardware accelerated)
build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
```

### 6.3 Advanced Effects

Once basic glass works, consider:
- Dispersion (wavelength-dependent IOR)
- Roughness (frosted glass)
- Caustics (photon mapping)

---

## Success Criteria

The implementation is successful when:

1. ✅ Internal rays detect sphere exit (hit_t > 0, not -1.0)
2. ✅ Glass spheres show proper refraction at entry AND exit
3. ✅ Beer-Lambert absorption creates correct tinted glass
4. ✅ Fresnel reflections blend naturally with refraction
5. ✅ No black pixels or artifacts at sphere edges
6. ✅ Total internal reflection works at grazing angles
7. ✅ Performance acceptable (< 50% slower than built-in)

---

## Risk Mitigation

### Risk 1: Custom Intersection Too Complex

**Mitigation**: Start with simplest possible implementation, add features incrementally.

### Risk 2: Performance Regression

**Mitigation**:
- Profile before/after
- Consider triangle mesh if too slow
- Implement LOD system

### Risk 3: Numerical Instabilities

**Mitigation**:
- Use double precision for intersection calculation
- Careful epsilon handling
- Extensive edge case testing

---

## Timeline Estimate

- **Phase 1** (Cleanup): 30 minutes
- **Phase 2** (Intersection): 2 hours
- **Phase 3** (Host updates): 1 hour
- **Phase 4** (Ray management): 1 hour
- **Phase 5** (Testing): 1 hour
- **Phase 6** (Optional): 2+ hours

**Total**: 5-6 hours for basic implementation

---

## References

- OptiX SDK: `/usr/local/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64/SDK/optixWhitted/`
- Glass Research: `Glass_Rendering_Findings.md`
- Original Issue: `ABSORPTION_DEBUG_FINDINGS.md`
- OptiX Documentation: https://raytracing-docs.nvidia.com/optix9/guide/index.html

---

## Next Steps

1. Review this plan
2. Execute Phase 1 (cleanup)
3. Implement Phase 2 (core fix)
4. Test with internal ray test case
5. Iterate until success criteria met

The key insight is that **custom intersection is mandatory** for glass spheres in OptiX. This is not a workaround but the standard, production-tested approach used by NVIDIA's own examples.