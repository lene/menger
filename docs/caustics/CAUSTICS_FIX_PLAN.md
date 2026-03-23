# Plan: Generalize PPM Caustics from Sphere-Specific to Arbitrary Geometry

## Context

The current caustics implementation produces correct results for a single glass sphere but is **hard-coded for sphere geometry** in 6 places. The physics core (Snell's law, Fresnel, Beer-Lambert, TIR, PPM algorithm, Gaussian kernel, grid acceleration, exponential tone mapping, screen blending) is already fully general and does not change.

The goal is to replace the manual `intersectSphere()` call in photon tracing with `optixTrace()` using a new `RAY_TYPE_PHOTON`, enabling caustics through any geometry type (spheres, triangle meshes, cylinders) with per-instance materials.

**Outcome:** Caustics work for arbitrary glass objects -- multiple spheres with different IORs, glass cubes (triangle mesh), colored glass, etc.

---

## What's Sphere-Specific (6 items to fix)

| # | Component | Location | Fix |
|---|-----------|----------|-----|
| 1 | Manual `intersectSphere()` in `tracePhoton()` | `caustics_ppm.cu:424-448, 629-668` | Replace with `optixTrace()` using `RAY_TYPE_PHOTON` |
| 2 | Normal from `normalize(hit - center)` | `caustics_ppm.cu:510` | Get normal from intersection attributes (already reported by existing intersection programs) |
| 3 | Global `params.sphere_ior` for refraction | `caustics_ppm.cu:519-520` | Per-instance `params.instance_materials[id].ior` |
| 4 | Global `params.sphere_color/scale` for absorption | `caustics_ppm.cu:463-468, 478` | Per-instance `params.instance_materials[id].color` |
| 5 | Emission targets `sphere_center/radius` | `caustics_ppm.cu:703-754` | Target bounding sphere of ALL glass objects |
| 6 | Hard-coded grid bounds +/-3.0 | `OptiXWrapper.cpp:740-745` | Dynamic bounds from scene AABB |

Plus: `bool hit_sphere` guard (line 643) becomes generic `bool has_refracted`.

---

## Step 1: Increase `numPayloadValues` from 4 to 6

**Est: 0.5h**

Photon closest-hit programs need 6 payload values to report geometry data. Current pipeline uses 4.

**File: `optix-jni/src/main/native/PipelineManager.cpp:19`**
```cpp
// BEFORE:
options.numPayloadValues = 4;
// AFTER:
options.numPayloadValues = 6;
```

**Verification:** All existing 2109 tests pass unchanged. No behavioral change.

---

## Step 2: Add `RAY_TYPE_PHOTON` to SBT, update stride from 2 to 3

**Est: 1.5h**

### 2a. Add constants

**File: `optix-jni/src/main/native/include/OptiXData.h:134-149`**

In `namespace SBTConstants`, change:
```cpp
constexpr unsigned int RAY_TYPE_PHOTON = 2;     // Photon tracing ray (caustics)
constexpr unsigned int MISS_PHOTON = 2;          // Photon ray miss shader index
constexpr unsigned int STRIDE_RAY_TYPES = 3;     // Was 2: primary + shadow + photon
```

### 2b. Fix hardcoded `* 2` in IAS SBT offset

**File: `optix-jni/src/main/native/OptiXWrapper.cpp:442`**
```cpp
// BEFORE:
oi.sbtOffset = inst.geometry_type * 2;
// AFTER:
oi.sbtOffset = inst.geometry_type * SBTConstants::STRIDE_RAY_TYPES;
```

All other references to `STRIDE_RAY_TYPES` in shader code (`helpers.cu`, `raygen_primary.cu`, `caustics_ppm.cu`) are symbolic and update automatically.

### 2c. Update SBT hitgroup record count from 6 to 9

**File: `optix-jni/src/main/native/PipelineManager.cpp:212-275`**

Update comment and record count:
```cpp
// SBT layout: [0]=sphere_primary, [1]=sphere_shadow, [2]=sphere_photon,
//             [3]=triangle_primary, [4]=triangle_shadow, [5]=triangle_photon,
//             [6]=cylinder_primary, [7]=cylinder_shadow, [8]=cylinder_photon
// Offset calculation: geometry_type * 3 + ray_type (0=primary, 1=shadow, 2=photon)
constexpr int num_records = 9;  // 3 geometry types * 3 ray types
```

Reindex all existing record assignments:
- Sphere primary: index 0 (was 0)
- Sphere shadow: index 1 (was 1)
- Sphere photon: index 2 (NEW)
- Triangle primary: index 3 (was 2)
- Triangle shadow: index 4 (was 3)
- Triangle photon: index 5 (NEW)
- Cylinder primary: index 6 (was 4)
- Cylinder shadow: index 7 (was 5)
- Cylinder photon: index 8 (NEW)

The photon hitgroup records (indices 2, 5, 8) will be populated in Step 4 after the programs are created.

### 2d. Update miss record count from 2 to 3

**File: `optix-jni/src/main/native/PipelineManager.cpp:184-210`**

Change miss record array size and count:
```cpp
MissSbtRecord miss_records[3];  // Was 2
// ... existing [0] and [1] setup ...
// [2] = photon miss (populated in Step 4)
sbt.missRecordCount = 3;  // Was 2
```

**Verification:** All existing tests pass. New ray type exists in SBT but no programs are registered yet (records will be populated in Step 4 -- use placeholder headers for now, or defer SBT creation until Step 4).

**Implementation note:** Steps 2c and 2d require placeholder program group headers for the new records. Either (a) defer actual SBT creation to after Step 4, or (b) create the records with null headers and fill them in Step 4. Option (a) is safer -- do Steps 2-4 as a single atomic change.

---

## Step 3: Write photon closest-hit and miss programs

**Est: 3h**

These programs are simpler than rendering closest-hit -- they only report geometry data back through payloads, no color computation.

### Payload convention for photon rays

| Payload | Content | Encoding |
|---------|---------|----------|
| p0 | outward_normal.x | `__float_as_uint` |
| p1 | outward_normal.y | `__float_as_uint` |
| p2 | outward_normal.z | `__float_as_uint` |
| p3 | hit distance t | `__float_as_uint` |
| p4 | instance_id | unsigned int |
| p5 | flags (MISS, GLASS) | bitmask |

Flag bits:
```cuda
constexpr unsigned int PHOTON_FLAG_MISS  = 0x1;  // No geometry hit
constexpr unsigned int PHOTON_FLAG_GLASS = 0x2;  // Hit glass (IOR > 1.01)
```

### 3a. `__closesthit__photon_sphere`

**File: `optix-jni/src/main/native/shaders/caustics_ppm.cu`** (add before `createONB`)

The sphere intersection program already reports outward normal via attributes 0-2 and entering/exiting via `hit_kind` (0=ENTER, 1=EXIT). Normals are in **object space** for IAS mode.

```cuda
extern "C" __global__ void __closesthit__photon_sphere() {
    // Normal from intersection attributes (object space in IAS, world space in single-object)
    float3 normal = make_float3(
        __uint_as_float(optixGetAttribute_0()),
        __uint_as_float(optixGetAttribute_1()),
        __uint_as_float(optixGetAttribute_2())
    );

    // Transform normal to world space if using IAS
    if (params.use_ias) {
        // optixTransformNormalFromObjectToWorldSpace handles non-uniform scale
        normal = normalize(optixTransformNormalFromObjectToWorldSpace(normal));
    }

    const float t = optixGetRayTmax();

    // Material lookup
    unsigned int instance_id = 0;
    bool is_glass = false;
    if (params.use_ias && params.instance_materials) {
        instance_id = optixGetInstanceId();
        is_glass = (params.instance_materials[instance_id].ior > 1.01f);
    } else {
        is_glass = (params.sphere_ior > 1.01f);
    }

    optixSetPayload_0(__float_as_uint(normal.x));
    optixSetPayload_1(__float_as_uint(normal.y));
    optixSetPayload_2(__float_as_uint(normal.z));
    optixSetPayload_3(__float_as_uint(t));
    optixSetPayload_4(instance_id);
    optixSetPayload_5(is_glass ? PHOTON_FLAG_GLASS : 0u);
}
```

### 3b. `__closesthit__photon_triangle`

**File: `optix-jni/src/main/native/shaders/caustics_ppm.cu`**

Triangle normals come from barycentric interpolation of vertex normals. The `TriangleHitGroupData` struct provides vertices and indices. The existing pattern from `hit_triangle.cu` can be reused.

```cuda
extern "C" __global__ void __closesthit__photon_triangle() {
    const TriangleHitGroupData* hit_data =
        reinterpret_cast<TriangleHitGroupData*>(optixGetSbtDataPointer());
    const float t = optixGetRayTmax();

    // Interpolate normal from vertex normals using barycentrics
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    const float2 bary = optixGetTriangleBarycentrics();
    const float u = bary.x;
    const float v = bary.y;
    const float w = 1.0f - u - v;

    const unsigned int idx0 = hit_data->indices[prim_idx * 3 + 0];
    const unsigned int idx1 = hit_data->indices[prim_idx * 3 + 1];
    const unsigned int idx2 = hit_data->indices[prim_idx * 3 + 2];
    const unsigned int stride = hit_data->vertex_stride;
    // Vertex layout: [pos.x, pos.y, pos.z, normal.x, normal.y, normal.z, ...]
    const float* v0 = &hit_data->vertices[idx0 * stride];
    const float* v1 = &hit_data->vertices[idx1 * stride];
    const float* v2 = &hit_data->vertices[idx2 * stride];

    float3 normal;
    if (stride >= 6) {
        // Has vertex normals at offset 3
        normal = normalize(make_float3(
            w * v0[3] + u * v1[3] + v * v2[3],
            w * v0[4] + u * v1[4] + v * v2[4],
            w * v0[5] + u * v1[5] + v * v2[5]
        ));
    } else {
        // Compute geometric normal from triangle edges
        const float3 e1 = make_float3(v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]);
        const float3 e2 = make_float3(v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]);
        normal = normalize(cross(e1, e2));
    }

    // Transform to world space if IAS
    if (params.use_ias) {
        normal = normalize(optixTransformNormalFromObjectToWorldSpace(normal));
    }

    unsigned int instance_id = 0;
    bool is_glass = false;
    if (params.use_ias && params.instance_materials) {
        instance_id = optixGetInstanceId();
        is_glass = (params.instance_materials[instance_id].ior > 1.01f);
    } else {
        is_glass = (hit_data->ior > 1.01f);
    }

    optixSetPayload_0(__float_as_uint(normal.x));
    optixSetPayload_1(__float_as_uint(normal.y));
    optixSetPayload_2(__float_as_uint(normal.z));
    optixSetPayload_3(__float_as_uint(t));
    optixSetPayload_4(instance_id);
    optixSetPayload_5(is_glass ? PHOTON_FLAG_GLASS : 0u);
}
```

### 3c. `__closesthit__photon_cylinder`

**File: `optix-jni/src/main/native/shaders/caustics_ppm.cu`**

Same pattern as sphere -- normal from attributes 0-2, material from instance_id.

```cuda
extern "C" __global__ void __closesthit__photon_cylinder() {
    float3 normal = make_float3(
        __uint_as_float(optixGetAttribute_0()),
        __uint_as_float(optixGetAttribute_1()),
        __uint_as_float(optixGetAttribute_2())
    );
    if (params.use_ias) {
        normal = normalize(optixTransformNormalFromObjectToWorldSpace(normal));
    }
    const float t = optixGetRayTmax();

    unsigned int instance_id = 0;
    bool is_glass = false;
    if (params.use_ias && params.instance_materials) {
        instance_id = optixGetInstanceId();
        is_glass = (params.instance_materials[instance_id].ior > 1.01f);
    }

    optixSetPayload_0(__float_as_uint(normal.x));
    optixSetPayload_1(__float_as_uint(normal.y));
    optixSetPayload_2(__float_as_uint(normal.z));
    optixSetPayload_3(__float_as_uint(t));
    optixSetPayload_4(instance_id);
    optixSetPayload_5(is_glass ? PHOTON_FLAG_GLASS : 0u);
}
```

### 3d. `__miss__photon`

**File: `optix-jni/src/main/native/shaders/caustics_ppm.cu`**

```cuda
extern "C" __global__ void __miss__photon() {
    optixSetPayload_5(PHOTON_FLAG_MISS);
}
```

---

## Step 4: Register photon programs in PipelineManager

**Est: 2.5h**

### 4a. Add program group members

**File: `optix-jni/src/main/native/include/PipelineManager.h`**

Add after existing caustics members:
```cpp
// Photon ray program groups (for caustics photon tracing)
OptixProgramGroup photon_sphere_hitgroup = nullptr;
OptixProgramGroup photon_triangle_hitgroup = nullptr;
OptixProgramGroup photon_cylinder_hitgroup = nullptr;
OptixProgramGroup photon_miss = nullptr;
```

### 4b. Create program groups

**File: `optix-jni/src/main/native/PipelineManager.cpp`** -- in `createProgramGroups()` after line 136:

```cpp
// Photon ray hit groups (caustics)
photon_sphere_hitgroup = optix_context.createHitgroupProgramGroup(
    module, "__closesthit__photon_sphere",
    module, "__intersection__sphere"  // Reuse existing sphere intersection
);
photon_triangle_hitgroup = optix_context.createTriangleHitgroupProgramGroup(
    module, "__closesthit__photon_triangle"
);
photon_cylinder_hitgroup = optix_context.createHitgroupProgramGroup(
    cylinder_module, "__closesthit__photon_cylinder",
    cylinder_module, "__intersection__cylinder"  // Reuse existing cylinder intersection
);
photon_miss = optix_context.createMissProgramGroup(
    module, "__miss__photon"
);
```

**Important:** The photon closest-hit programs live in `caustics_ppm.cu` which compiles into the main `module` (same PTX). Verify that `caustics_ppm.cu` is `#include`-ed into the main shader compilation unit, or compile it separately and use the correct module reference. Check `CMakeLists.txt` shader compilation setup.

### 4c. Update pipeline

**File: `optix-jni/src/main/native/PipelineManager.cpp:139-170`**

```cpp
constexpr int NUM_PROGRAM_GROUPS = 19;  // Was 15: +3 photon hitgroups + 1 photon miss
OptixProgramGroup program_groups[] = {
    // ... existing 15 ...
    photon_sphere_hitgroup,
    photon_triangle_hitgroup,
    photon_cylinder_hitgroup,
    photon_miss
};
```

### 4d. Populate SBT hitgroup records for photon ray type

**File: `optix-jni/src/main/native/PipelineManager.cpp`** -- in `createHitgroupRecords()`:

After existing records, add photon records at indices 2, 5, 8 (per the new SBT layout from Step 2c):

```cpp
// Sphere photon [2]
HitGroupSbtRecord* sphere_photon = reinterpret_cast<HitGroupSbtRecord*>(
    hitgroup_records + 2 * record_size);
optixSbtRecordPackHeader(photon_sphere_hitgroup, sphere_photon);
sphere_photon->data = sphere_data;

// Triangle photon [5]
TriangleHitGroupSbtRecord* tri_photon = reinterpret_cast<TriangleHitGroupSbtRecord*>(
    hitgroup_records + 5 * record_size);
optixSbtRecordPackHeader(photon_triangle_hitgroup, tri_photon);
if (scene.hasTriangleMesh()) {
    tri_photon->data = tri_data;
}

// Cylinder photon [8]
HitGroupSbtRecord* cylinder_photon = reinterpret_cast<HitGroupSbtRecord*>(
    hitgroup_records + 8 * record_size);
optixSbtRecordPackHeader(photon_cylinder_hitgroup, cylinder_photon);
cylinder_photon->data = sphere_data;  // Placeholder (cylinder uses instance data)
```

### 4e. Populate SBT photon miss record

In `createMissRecords()`:
```cpp
MissSbtRecord miss_records[3];
// ... existing [0] primary, [1] shadow ...
optixSbtRecordPackHeader(photon_miss, &miss_records[2]);
miss_records[2].data = ms_data;
sbt.missRecordCount = 3;
```

### 4f. Cleanup

In `cleanup()`, add:
```cpp
destroyProgramGroupIfExists(photon_sphere_hitgroup);
destroyProgramGroupIfExists(photon_triangle_hitgroup);
destroyProgramGroupIfExists(photon_cylinder_hitgroup);
destroyProgramGroupIfExists(photon_miss);
```

**Verification:** All existing tests pass. Photon ray type is fully wired but not yet called from `tracePhoton()`.

---

## Step 5: Rewrite `tracePhoton()` to use `optixTrace()`

**Est: 4h** -- This is the critical correctness step.

### 5a. Delete `intersectSphere()` (lines 424-448)

No longer needed -- OptiX handles all intersection.

### 5b. Rename `handlePhotonSphereHit()` to `handlePhotonGlassHit()`

**Change signature** from:
```cuda
__device__ bool handlePhotonGlassHit(
    float3& origin, float3& dir, float3& flux,
    float t, const float3& sphere_center)
```
to:
```cuda
__device__ bool handlePhotonGlassHit(
    float3& origin, float3& dir, float3& flux,
    float t, const float3& outward_normal,
    const InstanceMaterial& mat)
```

**Changes inside the function:**
- Line 510: Delete `normalize(hit_point - sphere_center)`. Use `outward_normal` parameter directly.
- Line 516: `entering = dot(dir, outward_normal) < 0.0f` -- unchanged (already general).
- Line 519-520: Replace `params.sphere_ior` with `mat.ior`.
- Line 530: Change `applyPhotonBeerLambert(flux, t, entering)` to `applyPhotonBeerLambert(flux, t, entering, mat)`.

### 5c. Modify `applyPhotonBeerLambert()` signature

**Change** from:
```cuda
__device__ void applyPhotonBeerLambert(float3& flux, float distance, bool entering)
```
to:
```cuda
__device__ void applyPhotonBeerLambert(float3& flux, float distance, bool entering,
                                        const InstanceMaterial& mat)
```

**Changes inside:**
- Lines 463-465: Replace `params.sphere_color[0-2]` with `mat.color[0-2]`.
- Line 468: Replace `params.sphere_color[3]` with `mat.color[3]`.
- Line 478: Replace `params.sphere_scale` with `1.0f` (world-space units; scale is already handled by instance transform).

### 5d. Rewrite `tracePhoton()`

Replace lines 629-668 entirely:

```cuda
__device__ void tracePhoton(
    float3 origin, float3 dir, float3 flux,
    unsigned int& seed, int max_bounces
) {
    bool has_refracted = false;

    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0;
        optixTrace(
            params.handle,
            origin, dir,
            CONTINUATION_RAY_OFFSET,   // tmin
            MAX_RAY_DISTANCE,          // tmax
            0.0f,                      // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            params.sbt_base_offset + SBTConstants::RAY_TYPE_PHOTON,
            SBTConstants::STRIDE_RAY_TYPES,
            SBTConstants::MISS_PHOTON,
            p0, p1, p2, p3, p4, p5
        );

        // Miss -- photon escaped scene
        if (p5 & PHOTON_FLAG_MISS) {
            // After refraction, try depositing on planes (planes are not in IAS)
            if (has_refracted) {
                checkPlaneIntersection(origin, dir, flux);
            }
            if (params.caustics.stats && bounce == 0) {
                atomicAdd(&params.caustics.stats->sphere_misses, 1ULL);
            }
            return;
        }

        // Unpack hit data
        const float3 outward_normal = make_float3(
            __uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
        const float t = __uint_as_float(p3);
        const unsigned int instance_id = p4;
        const bool is_glass = (p5 & PHOTON_FLAG_GLASS) != 0;

        if (is_glass) {
            has_refracted = true;
            if (params.caustics.stats) {
                atomicAdd(&params.caustics.stats->sphere_hits, 1ULL);
            }
            // Get per-instance material
            if (params.use_ias && params.instance_materials) {
                handlePhotonGlassHit(origin, dir, flux, t, outward_normal,
                                     params.instance_materials[instance_id]);
            } else {
                // Single-object fallback: build material from global params
                InstanceMaterial mat = {};
                mat.color[0] = params.sphere_color[0];
                mat.color[1] = params.sphere_color[1];
                mat.color[2] = params.sphere_color[2];
                mat.color[3] = params.sphere_color[3];
                mat.ior = params.sphere_ior;
                handlePhotonGlassHit(origin, dir, flux, t, outward_normal, mat);
            }
        } else {
            // Hit diffuse surface -- deposit if photon has been refracted
            if (has_refracted) {
                const float3 hit_pos = origin + dir * t;
                depositPhoton(hit_pos, dir, flux);
            }
            return;
        }
    }
}
```

**Design note:** Planes are NOT in the IAS (they are rendered via the miss shader). After a photon misses all geometry post-refraction, `checkPlaneIntersection()` handles plane deposition. This maintains backward compatibility. When any geometry in the IAS is hit and is diffuse (`!is_glass`), that surface also receives caustics.

### 5e. Update `handlePhotonGlassHit()` internals

Remove the `sphere_center` parameter and sphere-specific stats. The Fresnel, Snell's law, TIR, and Beer-Lambert code stays identical except for the data sources noted in 5b/5c.

**Verification:** This is the critical regression checkpoint.
- CausticsReferenceSuite: brightness ratio should be within ~10% of pre-refactor (38.3% of PBRT reference).
- CausticsValidationSuite: C1-C5 tests pass.
- All 2109 tests pass.

---

## Step 6: Generalize photon emission to target glass AABB

**Est: 2h**

### 6a. Update `CausticsParams` data structure

**File: `optix-jni/src/main/native/include/OptiXData.h:343-345`**

Replace:
```cpp
float sphere_center[3];   // DELETE
float sphere_radius;       // DELETE
```
With:
```cpp
float glass_aabb_center[3];   // Bounding sphere center of all glass objects
float glass_aabb_radius;       // Bounding sphere radius enclosing all glass objects
```

### 6b. Compute glass AABB on host

**File: `optix-jni/src/main/native/OptiXWrapper.cpp:731-732`**

Replace the direct `sphere_center`/`sphere_radius` copy with:
```cpp
// Compute bounding sphere of all glass instances
float glass_min[3] = { 1e16f,  1e16f,  1e16f};
float glass_max[3] = {-1e16f, -1e16f, -1e16f};
bool has_glass = false;

if (impl->use_ias) {
    for (const auto& inst : impl->instances) {
        if (!inst.active || inst.ior <= 1.01f) continue;
        has_glass = true;
        // Extract position from 4x3 row-major transform (column 3)
        float cx = inst.transform[3];
        float cy = inst.transform[7];
        float cz = inst.transform[11];
        // Estimate instance radius from transform scale
        float sx = sqrtf(inst.transform[0]*inst.transform[0] +
                         inst.transform[1]*inst.transform[1] +
                         inst.transform[2]*inst.transform[2]);
        float sy = sqrtf(inst.transform[4]*inst.transform[4] +
                         inst.transform[5]*inst.transform[5] +
                         inst.transform[6]*inst.transform[6]);
        float sz = sqrtf(inst.transform[8]*inst.transform[8] +
                         inst.transform[9]*inst.transform[9] +
                         inst.transform[10]*inst.transform[10]);
        float r = std::max({sx, sy, sz});
        for (int d = 0; d < 3; ++d) {
            float c = (d==0) ? cx : (d==1) ? cy : cz;
            glass_min[d] = std::min(glass_min[d], c - r);
            glass_max[d] = std::max(glass_max[d], c + r);
        }
    }
}
if (!has_glass) {
    // Fallback: single-object mode
    std::memcpy(params.caustics.glass_aabb_center, sphere.center, 3 * sizeof(float));
    params.caustics.glass_aabb_radius = sphere.radius;
} else {
    for (int d = 0; d < 3; ++d)
        params.caustics.glass_aabb_center[d] = (glass_min[d] + glass_max[d]) * 0.5f;
    float dx = glass_max[0]-glass_min[0], dy = glass_max[1]-glass_min[1], dz = glass_max[2]-glass_min[2];
    params.caustics.glass_aabb_radius = sqrtf(dx*dx + dy*dy + dz*dz) * 0.5f;
}
```

### 6c. Update photon emission functions

**File: `optix-jni/src/main/native/shaders/caustics_ppm.cu`**

In `emitPointPhoton()` (lines 744-754) and `emitDirectionalPhoton()` (lines 703-715), replace all references:
- `params.caustics.sphere_center[0-2]` -> `params.caustics.glass_aabb_center[0-2]`
- `params.caustics.sphere_radius` -> `params.caustics.glass_aabb_radius`

This is a mechanical find-and-replace within these two functions.

**Verification:** CausticsReferenceSuite passes (single sphere scene -- AABB of one sphere = that sphere's bounds). Add multi-sphere test (Step 9).

---

## Step 7: Dynamic grid bounds from scene AABB

**Est: 1h**

**File: `optix-jni/src/main/native/OptiXWrapper.cpp:739-755`**

Replace hardcoded +/-3.0 with dynamic computation:
```cpp
// Compute scene AABB for caustics grid (include all instances + planes)
float scene_min[3] = { 1e16f,  1e16f,  1e16f};
float scene_max[3] = {-1e16f, -1e16f, -1e16f};

if (impl->use_ias) {
    for (const auto& inst : impl->instances) {
        if (!inst.active) continue;
        float cx = inst.transform[3], cy = inst.transform[7], cz = inst.transform[11];
        // ... same scale estimation as Step 6b ...
        // Expand scene bounds by instance bounds
    }
} else {
    for (int d = 0; d < 3; ++d) {
        scene_min[d] = sphere.center[d] - sphere.radius;
        scene_max[d] = sphere.center[d] + sphere.radius;
    }
}

// Include planes in bounds
for (int i = 0; i < impl->config.getNumPlanes(); ++i) {
    const auto& plane = impl->config.getPlanes()[i];
    if (!plane.enabled) continue;
    scene_min[plane.axis] = std::min(scene_min[plane.axis], plane.value);
    scene_max[plane.axis] = std::max(scene_max[plane.axis], plane.value);
}

// Pad by 2 * initial_radius to ensure all hit points are within grid
float padding = params.caustics.initial_radius * 2.0f;
for (int d = 0; d < 3; ++d) {
    params.caustics.grid_min[d] = scene_min[d] - padding;
    params.caustics.grid_max[d] = scene_max[d] + padding;
}
```

**Verification:** Existing tests pass. Scene with objects far from origin produces caustics.

---

## Step 8: Update documentation

**Est: 0.5h**

**File: `docs/caustics/CAUSTICS_ANALYSIS.md`** -- Update "Sphere-Specific Limitations" section to mark as resolved.

**File: `docs/caustics/CAUSTICS_ITERATION_LOG.md`** -- Add entry for geometry generalization.

---

## Step 9: New test cases

**Est: 3h**

All new tests go in existing test files.

### Test 1: Regression -- single sphere brightness unchanged

**File: `CausticsReferenceSuite.scala`**

The existing brightness comparison test (`should compare caustic brightness within 40% of reference`) serves as the regression test. Verify it still passes with ratio >= 35%.

### Test 2: Two glass spheres with different IOR

**File: `CausticsValidationSuite.scala`**

```scala
it should "produce caustics from multiple glass spheres" taggedAs (Slow) in {
  renderer.enableIAS()
  renderer.addSphereInstance(
    Vector[3](-1.5f, 0.0f, 0.0f),
    Material(color = Color.White, ior = 1.5f),  // glass
    scale = 1.0f
  )
  renderer.addSphereInstance(
    Vector[3](1.5f, 0.0f, 0.0f),
    Material(color = Color.White, ior = 1.33f),  // water
    scale = 1.0f
  )
  // Add plane, lights, camera...
  renderer.enableCaustics(photonsPerIter = 10000, iterations = 3)
  val result = renderer.renderWithStats(ImageSize(400, 300))

  // Detect caustic regions in left and right halves of floor
  val leftBrightness = regionBrightness(result.image, size, 50, 200, 100, 100)
  val rightBrightness = regionBrightness(result.image, size, 250, 200, 100, 100)
  val bgBrightness = regionBrightness(result.image, size, 150, 250, 100, 50)

  // Both spheres should produce caustics brighter than background
  leftBrightness should be > bgBrightness
  rightBrightness should be > bgBrightness
}
```

### Test 3: Colored glass -- Beer-Lambert per-instance

**File: `CausticsValidationSuite.scala`**

```scala
it should "produce color-tinted caustics from colored glass" taggedAs (Slow) in {
  // Red glass sphere: only red light passes through
  renderer.enableIAS()
  renderer.addSphereInstance(
    Vector[3](0.0f, 0.0f, 0.0f),
    Material(color = Color(1.0f, 0.2f, 0.2f, 0.5f), ior = 1.5f),
    scale = 1.0f
  )
  // ... plane, lights, camera, caustics ...
  val result = renderer.renderWithStats(ImageSize(400, 300))

  // Measure R, G, B channels separately in caustic region
  val (cx, cy, cw, ch) = detectCausticRegion(result.image, size)
  val rAvg = channelBrightness(result.image, size, cx, cy, cw, ch, channel = 0)
  val bAvg = channelBrightness(result.image, size, cx, cy, cw, ch, channel = 2)

  // Red channel should be significantly brighter than blue (Beer-Lambert absorbs blue)
  rAvg should be > (bAvg * 1.5)
}
```

### Test 4: No glass -- graceful no-op

**File: `CausticsValidationSuite.scala`**

```scala
it should "render without error when no glass objects exist" taggedAs (Slow) in {
  renderer.enableIAS()
  renderer.addSphereInstance(
    Vector[3](0.0f, 0.0f, 0.0f),
    Material(color = Color.White, ior = 1.0f),  // opaque, not glass
    scale = 1.0f
  )
  // Enable caustics anyway
  renderer.enableCaustics(photonsPerIter = 1000, iterations = 1)
  val result = renderer.renderWithStats(ImageSize(400, 300))

  // Should complete without error; no caustic brightness above background
  val bgBrightness = regionBrightness(result.image, size, 100, 200, 200, 100)
  // Just verify it rendered (non-zero image)
  bgBrightness should be > 0.0
}
```

### Test 5: Triangle mesh glass (if mesh IOR support exists)

**File: `CausticsValidationSuite.scala`**

```scala
it should "produce caustics from glass triangle mesh" taggedAs (Slow) in {
  // Glass cube via triangle mesh with IOR 1.5
  // Use existing cube mesh builder with glass material
  renderer.enableIAS()
  renderer.addTriangleMeshInstance(
    cubeMesh, transform, Material(color = Color.White, ior = 1.5f))
  // ... plane, lights, caustics ...
  val result = renderer.renderWithStats(ImageSize(400, 300))

  // Verify caustic exists (brightness above background)
  val causticBrightness = regionBrightness(result.image, size, ...)
  val bgBrightness = regionBrightness(result.image, size, ...)
  causticBrightness should be > bgBrightness
}
```

---

## Files Modified -- Summary

| File | Changes |
|------|---------|
| `optix-jni/src/main/native/include/OptiXData.h` | Add `RAY_TYPE_PHOTON`, `MISS_PHOTON`; change `STRIDE_RAY_TYPES` 2->3; replace `sphere_center/radius` with `glass_aabb_center/radius` in `CausticsParams` |
| `optix-jni/src/main/native/include/PipelineManager.h` | Add 4 program group members |
| `optix-jni/src/main/native/PipelineManager.cpp` | `numPayloadValues` 4->6; register 4 programs; `NUM_PROGRAM_GROUPS` 15->19; SBT 6->9 hitgroups, 2->3 miss; cleanup |
| `optix-jni/src/main/native/OptiXWrapper.cpp` | Fix hardcoded `*2`; compute glass AABB; compute dynamic grid bounds |
| `optix-jni/src/main/native/shaders/caustics_ppm.cu` | Add 3 closest-hit + 1 miss programs; delete `intersectSphere()`; rewrite `tracePhoton()`; rename+modify `handlePhotonGlassHit()`; modify `applyPhotonBeerLambert()` |
| `optix-jni/src/test/scala/.../CausticsValidationSuite.scala` | Add tests C9-C12 (multi-sphere, colored glass, no-glass, triangle mesh) |
| `docs/caustics/CAUSTICS_FIX_PLAN.md` | Replace with this plan |
| `docs/caustics/CAUSTICS_ANALYSIS.md` | Update status |

---

## Time Estimates

| Step | Description | Est. Hours |
|------|-------------|-----------|
| 1 | Increase numPayloadValues 4->6 | 0.5 |
| 2 | Add RAY_TYPE_PHOTON, stride 2->3, SBT 6->9 | 1.5 |
| 3 | Write 3 photon closest-hit + 1 miss program | 3.0 |
| 4 | Register programs in PipelineManager + SBT | 2.5 |
| 5 | Rewrite tracePhoton() with optixTrace() | 4.0 |
| 6 | Generalize emission to glass AABB | 2.0 |
| 7 | Dynamic grid bounds | 1.0 |
| 8 | Documentation | 0.5 |
| 9 | New tests | 3.0 |
| **Total** | | **~18h** |

Steps 1-4 should be committed together (SBT change is atomic).
Step 5 is a separate commit (critical correctness checkpoint).
Steps 6-7 are incremental commits.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **SBT stride 2->3 breaks all rendering** | All `optixTrace()` calls use wrong offsets | All shader-side calls use `SBTConstants::STRIDE_RAY_TYPES` symbolically -- only the one hardcoded `*2` in `OptiXWrapper.cpp:442` needs manual fix. Grep for `* 2` near `sbtOffset` to verify. |
| **Object-space vs world-space normals** | Wrong refraction angles in IAS mode | Photon closest-hit must call `optixTransformNormalFromObjectToWorldSpace()` for IAS mode (included in code above). |
| **Performance regression** | OptiX trace slower than manual intersection for single sphere | Acceptable tradeoff. Photon tracing phase is not the bottleneck (grid+radiance phases dominate). OptiX BVH is faster for multi-object. |
| **Planes not in IAS** | Photons miss planes after refraction | `checkPlaneIntersection()` fallback after miss preserves current behavior. |
| **Single-object mode (non-IAS) compatibility** | `optixGetInstanceId()` undefined in non-IAS | Closest-hit checks `params.use_ias` and falls back to global `params.sphere_ior`/`sphere_color`. `tracePhoton()` also has this fallback. |
| **numPayloadValues increase** | Register pressure -> lower occupancy | 2 extra registers per program is negligible. Can verify with `--ptxas-options=-v`. |

---

## Verification Plan

After each step:
```bash
sbt test    # All 2109+ tests pass
```

After Step 5 (critical checkpoint):
```bash
sbt "project optixJni" "testOnly menger.optix.caustics.*"   # Caustics-specific tests
```

After all steps:
```bash
# Visual verification
sbt "project mengerApp" "run --scene examples.dsl.CausticsDemo --optix --caustics --save /tmp/caustics-general.png"
# Compare /tmp/caustics-general.png with pre-refactor output
```

## Deferred (out of scope)

- **Hit point generation on arbitrary diffuse surfaces** -- currently only `planes[0]` collects hit points. Generalizing this to any diffuse surface hit by camera rays requires its own design. Add TODO comment in `__raygen__hitpoints`.
- **Multiple light support** -- current code uses `lights[0]` only for photon emission. Not blocked by geometry generalization.
- **Dynamic grid from GPU reduction** -- compute AABB on GPU instead of CPU. Optimization, not correctness.
