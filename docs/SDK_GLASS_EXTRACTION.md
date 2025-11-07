# SDK Glass Sphere Implementation - Extraction Guide

**Date:** November 7, 2025
**Goal:** Extract NVIDIA SDK's working glass sphere implementation

## Source Files

**From `/usr/local/optix/SDK/cuda/`:**
- `geometry.cu` - `__intersection__sphere_shell()` (lines ~100-180)
- `shading.cu` - `__closesthit__glass_radiance()` (lines 229-328)
- `sphere.cu` - `__intersection__sphere()` (lines 40-100) - simpler solid sphere
- `GeometryData.h` - Data structures
- `whitted.h` - Constants and payload structures

## Phase 1: Hollow Sphere (SDK As-Is)

### Step 1: Data Structures Needed

**From GeometryData.h:**
```cpp
struct SphereShell {
    enum HitType {
        HIT_OUTSIDE_FROM_OUTSIDE = 1u << 0,  // Entry from outside
        HIT_OUTSIDE_FROM_INSIDE  = 1u << 1,  // Exit to outside
        HIT_INSIDE_FROM_OUTSIDE  = 1u << 2,  // Entry to inner void
        HIT_INSIDE_FROM_INSIDE   = 1u << 3   // Exit from inner void
    };
    float3 center;
    float  radius1;  // Inner radius (hollow part)
    float  radius2;  // Outer radius
};
```

**Glass material (from MaterialData.h):**
```cpp
struct Glass {
    float3 extinction_constant;    // Beer-Lambert absorption (per wavelength)
    float3 refraction_color;
    float3 reflection_color;
    float  refraction_index;       // IOR
    int    refraction_maxdepth;
    int    reflection_maxdepth;
    float  importance_cutoff;
    float  cutoff_color;
    float  fresnel_exponent;
    float  fresnel_minimum;
    float  fresnel_maximum;
    float  shadow_attenuation;
};
```

### Step 2: Key SDK Implementation Details

**Beer-Lambert Application (shading.cu:271-280):**
```cuda
float3 beer_attenuation;
if( dot( n, ray_dir ) > 0 )  // Exiting (ray pointing away from normal)
{
    beer_attenuation = exp( glass.extinction_constant * ray_t );
}
else  // Entering
{
    beer_attenuation = make_float3( 1 );  // No absorption on entry
}

// Later (line 324):
result = result * beer_attenuation;  // Apply to ENTIRE result
```

**Critical insight:** `ray_t` varies by pixel!
- Center: Longer path → larger `ray_t` → more absorption → darker
- Edges: Shorter path → smaller `ray_t` → less absorption → lighter

**Fresnel + Refraction (shading.cu:282-307):**
```cuda
// Compute refraction direction
if( refract( t, ray_dir, n, glass.refraction_index ) )
{
    float cos_theta = dot( ray_dir, n );
    if( cos_theta < 0.0f )
        cos_theta = -cos_theta;  // Entry
    else
        cos_theta = dot( t, n );  // Exit

    reflection = fresnel_schlick( cos_theta, ... );

    // Trace refracted ray from back hit point
    color = traceRadianceRay( bhp, t, depth + 1, importance );
    result += ( 1.0f - reflection ) * glass.refraction_color * color;
}

// Reflection (shading.cu:309-322)
r = reflect( ray_dir, n );
color = traceRadianceRay( fhp, r, depth + 1, importance );
result += reflection * glass.reflection_color * color;

result = result * beer_attenuation;  // Apply absorption to combined result
```

**Front/Back Hit Points (shading.cu:248-264):**
```cuda
float3 hit_point = ray_orig + ray_t * ray_dir;
float3 front_hit_point = hit_point, back_hit_point = hit_point;

// Offset by scene_epsilon to avoid self-intersection
if( hit_type & HIT_OUTSIDE_FROM_OUTSIDE || hit_type & HIT_INSIDE_FROM_INSIDE )
{
    front_hit_point += params.scene_epsilon * object_normal;
    back_hit_point -= params.scene_epsilon * object_normal;
}
else
{
    front_hit_point -= params.scene_epsilon * object_normal;
    back_hit_point += params.scene_epsilon * object_normal;
}

// Transform to world space
const float3 fhp = optixTransformPointFromObjectToWorldSpace( front_hit_point );
const float3 bhp = optixTransformPointFromObjectToWorldSpace( back_hit_point );
```

## Phase 2: Adapt to Solid Sphere

### Option A: Use SDK's solid sphere intersection

**From `/usr/local/optix/SDK/cuda/sphere.cu`:**
- Already has solid sphere intersection
- Simpler than sphere shell (no inner/outer logic)
- Just reports entry and exit hits

### Option B: Simplify sphere shell to solid

Set `radius1 = 0` → solid sphere (no hollow center)

**Modified hit types needed:**
- Entry: `HIT_OUTSIDE_FROM_OUTSIDE` only
- Exit: `HIT_OUTSIDE_FROM_INSIDE` only

### Glass Shader Adaptation

SDK's glass shader should work unchanged because:
1. Beer-Lambert uses `ray_t` which is automatically correct
2. Fresnel/refraction logic doesn't depend on hollow vs solid
3. Just need to handle entry/exit hit types correctly

## Our Current Data Structures

**OptiXData.h:**
```cpp
struct HitGroupData {
    float sphere_center[3];
    float sphere_radius;
    float sphere_color[4];  // RGBA
    float light_dir[3];
    float light_intensity;
    float ior;
    float scale;
};
```

**Mapping to SDK:**
- `sphere_center` → `SphereShell.center`
- `sphere_radius` → `SphereShell.radius2`
- `0.0f` → `SphereShell.radius1` (solid sphere)
- `ior` → `Glass.refraction_index`
- `sphere_color` → Needs to be split into:
  - RGB → `Glass.extinction_constant` (via `-log(rgb) * (1-alpha)`)
  - `Glass.refraction_color = make_float3(1,1,1)`
  - `Glass.reflection_color = make_float3(1,1,1)`

## Implementation Steps

### Step 1: Copy SDK Files

1. Copy sphere shell intersection to `sphere_combined.cu`
2. Copy glass closest hit shader to `sphere_combined.cu`
3. Extract needed helper functions (`fresnel_schlick`, `refract`, etc.)

### Step 2: Adapt Data Structures

1. Add `SphereShell` to `OptiXData.h`
2. Add `Glass` material to `OptiXData.h`
3. Update `HitGroupData` to include both

### Step 3: Update Host Code (OptiXWrapper.cpp)

1. Set up sphere shell geometry (radius1=0, radius2=sphere_radius)
2. Convert our sphere color → glass material parameters
3. Update SBT to include glass material

### Step 4: Test

1. Build and test hollow sphere (radius1 > 0) first
2. Then test solid sphere (radius1 = 0)
3. Verify varying absorption (center darker than edges)

## Expected Results

**Hollow sphere (radius1=0.4, radius2=0.5):**
- Should see through hollow center
- Glass shell should show varying absorption
- Edges lighter, thicker parts darker

**Solid sphere (radius1=0.0, radius2=0.5):**
- No hollow center
- Entire sphere shows glass effect
- Center (longest path) darkest
- Edges (shortest path) lightest

## Success Criteria

1. ✅ No solid color - varying absorption visible
2. ✅ Center darker than edges (Beer-Lambert working)
3. ✅ Fresnel reflection at grazing angles
4. ✅ Proper refraction (background distorted through sphere)
5. ✅ Color tint based on `sphere_color` RGB
6. ✅ Performance acceptable (~500+ fps)

## Files to Modify

1. `optix-jni/src/main/native/shaders/sphere_combined.cu`
   - Replace custom intersection with SDK sphere shell
   - Replace closest hit with SDK glass shader

2. `optix-jni/src/main/native/include/OptiXData.h`
   - Add SphereShell struct
   - Add Glass material struct
   - Update HitGroupData

3. `optix-jni/src/main/native/OptiXWrapper.cpp`
   - Update geometry setup for sphere shell
   - Convert sphere_color to glass material parameters
   - Update SBT

## Notes

- SDK uses `params.scene_epsilon` for ray offsetting (we use `Constants::CONTINUATION_RAY_OFFSET`)
- SDK uses payload struct `PayloadRadiance` (we use separate uint payloads)
- SDK traces to arbitrary depth (we limit to MAX_DEPTH=2)
- SDK has importance culling (we don't need this initially)
