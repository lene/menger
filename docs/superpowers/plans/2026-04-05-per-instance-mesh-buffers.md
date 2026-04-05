# Per-Instance Mesh Buffer Pointer Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix CUDA error 700 when rendering scenes with mixed triangle mesh types (e.g. sponge-volume + sponge-surface) in IAS multi-object mode, by giving each instance access to its own mesh's vertex/index buffers instead of reading from a single SBT record.

**Architecture:** Currently, the closest-hit shader reads vertex/index buffer pointers from the SBT's `TriangleHitGroupData`, which stores ONE mesh's data. In multi-mesh IAS mode, different instances reference different GAS handles with different buffers, but the shader always reads from the SBT buffer. When a ray hits mesh B (large), the primitive ID indexes into mesh A's (small) buffer — out-of-bounds read — CUDA error 700. Fix: store per-mesh buffer pointers in `InstanceMaterial` (already per-instance), populate them in `buildIAS()` from `triangle_meshes[mesh_index]`, and update `getTriangleGeometry()` to read from instance data in IAS mode.

**Tech Stack:** C++17, CUDA, OptiX 9.0, Scala 3 (test layer)

**Key constraint:** Single-object mode (non-IAS) must continue working unchanged — it reads from `TriangleHitGroupData` in the SBT. The new per-instance path is IAS-only.

---

### Task 1: Add `mesh_index` to `ObjectInstance` and store it in `addTriangleMeshInstance`

**Files:**
- Modify: `optix-jni/src/main/native/OptiXWrapper.cpp:59-72` (ObjectInstance struct)
- Modify: `optix-jni/src/main/native/OptiXWrapper.cpp:1020-1022` (addTriangleMeshInstance mesh_index assignment)

- [ ] **Step 1.1: Add `mesh_index` field to `ObjectInstance`**

In `optix-jni/src/main/native/OptiXWrapper.cpp`, find the `ObjectInstance` struct inside the `Impl` struct (around line 59). Add a `size_t mesh_index` field. Use `SIZE_MAX` as the sentinel value for non-triangle instances (spheres, cylinders):

```cpp
struct ObjectInstance {
    GeometryType geometry_type;           // Sphere or Triangle mesh
    OptixTraversableHandle gas_handle;    // GAS for this geometry type
    float transform[12];                  // 4x3 row-major transform matrix
    float color[4];                       // RGBA material color
    float ior;                            // Index of refraction
    float roughness;                      // 0=mirror, 1=diffuse (default: 0.5)
    float metallic;                       // 0=dielectric, 1=metal (default: 0.0)
    float specular;                       // Specular intensity (default: 0.5)
    float emission;                       // Emission intensity (default: 0.0)
    float film_thickness;                 // Thin-film thickness in nm (0 = none)
    int texture_index;                    // Index into textures array (-1 = no texture)
    bool active;                          // True if instance is enabled
    size_t mesh_index;                    // Index into triangle_meshes vector (SIZE_MAX = not a triangle)
};
```

- [ ] **Step 1.2: Store mesh_index in `addTriangleMeshInstance()`**

In `addTriangleMeshInstance()` (around line 1020-1022), the code already computes `mesh_index` as `impl->triangle_meshes.size() - 1`. After `inst.active = true;` (line 1047), add:

```cpp
    inst.mesh_index = mesh_index;
```

This stores which mesh entry this instance references, so `buildIAS()` can later look up the correct vertex/index buffers.

- [ ] **Step 1.3: Initialize mesh_index in `addSphereInstance()`**

In `addSphereInstance()` (around line 979), after `inst.active = true;`, add:

```cpp
    inst.mesh_index = SIZE_MAX;  // Not a triangle mesh instance
```

- [ ] **Step 1.4: Initialize mesh_index in `addCylinderInstance()`**

In `addCylinderInstance()` (find the function after addTriangleMeshInstance, around line 1062+), after `inst.active = true;`, add:

```cpp
    inst.mesh_index = SIZE_MAX;  // Not a triangle mesh instance
```

- [ ] **Step 1.5: Compile to verify**

Run: `sbt "project optixJni" nativeCompile`
Expected: Compiles successfully. No behavioral change yet — mesh_index is stored but not read.

---

### Task 2: Add per-instance buffer pointer fields to `InstanceMaterial`

**Files:**
- Modify: `optix-jni/src/main/native/include/OptiXData.h:162-174` (InstanceMaterial struct)

- [ ] **Step 2.1: Add vertex/index buffer fields to `InstanceMaterial`**

In `optix-jni/src/main/native/include/OptiXData.h`, find the `InstanceMaterial` struct (line 162-174). Add three new fields at the end, before the closing brace:

```cpp
// Per-instance material data for IAS (indexed by instance ID)
// Stored in GPU array, accessed via optixGetInstanceId()
struct InstanceMaterial {
    float color[4];             // RGBA color (alpha: 0=transparent, 1=opaque)
    float ior;                  // Index of refraction
    float roughness;            // 0=mirror, 1=diffuse (default: 0.5)
    float metallic;             // 0=dielectric, 1=metal (default: 0.0)
    float specular;             // Specular intensity (default: 0.5)
    float emission;             // Emission intensity (0.0-10.0, default: 0.0)
    unsigned int geometry_type; // GeometryType enum value
    int texture_index;          // Index into Params.textures array (-1 = no texture)
    float film_thickness;       // Thin-film thickness in nm (0 = no thin-film interference)
    // Per-mesh triangle buffer pointers (populated for triangle instances in IAS mode)
    float* vertices;            // Device pointer to vertex data (nullptr for non-triangle)
    unsigned int* indices;      // Device pointer to index data (nullptr for non-triangle)
    unsigned int vertex_stride; // Floats per vertex: 6 (pos+normal), 8 (+uv), 9 (+alpha)
};
```

- [ ] **Step 2.2: Compile to verify**

Run: `sbt "project optixJni" nativeCompile`
Expected: Compiles successfully. The new fields exist but are not populated yet (zero-initialized in buildIAS).

---

### Task 3: Populate per-instance buffer pointers in `buildIAS()`

**Files:**
- Modify: `optix-jni/src/main/native/OptiXWrapper.cpp:503-514` (buildIAS material loop)

- [ ] **Step 3.1: Add per-instance buffer pointers in the `buildIAS()` material population loop**

In `optix-jni/src/main/native/OptiXWrapper.cpp`, find the `buildIAS()` method (line 457). Inside the loop that builds `InstanceMaterial` entries (around line 503-514), after `mat.film_thickness = inst.film_thickness;`, add code to populate the new buffer pointer fields:

Replace this section (lines 503-514):
```cpp
        // Build material entry with PBR properties
        InstanceMaterial mat = {};
        std::memcpy(mat.color, inst.color, 4 * sizeof(float));
        mat.ior = inst.ior;
        mat.roughness = inst.roughness;
        mat.metallic = inst.metallic;
        mat.specular = inst.specular;
        mat.emission = inst.emission;
        mat.geometry_type = inst.geometry_type;
        mat.texture_index = inst.texture_index;
        mat.film_thickness = inst.film_thickness;
        materials.push_back(mat);
```

With:
```cpp
        // Build material entry with PBR properties
        InstanceMaterial mat = {};
        std::memcpy(mat.color, inst.color, 4 * sizeof(float));
        mat.ior = inst.ior;
        mat.roughness = inst.roughness;
        mat.metallic = inst.metallic;
        mat.specular = inst.specular;
        mat.emission = inst.emission;
        mat.geometry_type = inst.geometry_type;
        mat.texture_index = inst.texture_index;
        mat.film_thickness = inst.film_thickness;

        // Per-mesh triangle buffer pointers for IAS mode
        // Triangle instances store the mesh_index referencing triangle_meshes vector
        if (inst.geometry_type == GEOMETRY_TYPE_TRIANGLE
            && inst.mesh_index < impl->triangle_meshes.size()) {
            const auto& mesh = impl->triangle_meshes[inst.mesh_index];
            mat.vertices = reinterpret_cast<float*>(mesh.d_vertices);
            mat.indices = reinterpret_cast<unsigned int*>(mesh.d_indices);
            mat.vertex_stride = mesh.vertex_stride;
        } else {
            mat.vertices = nullptr;
            mat.indices = nullptr;
            mat.vertex_stride = 0;
        }

        materials.push_back(mat);
```

- [ ] **Step 3.2: Compile to verify**

Run: `sbt "project optixJni" nativeCompile`
Expected: Compiles successfully. Per-instance buffer pointers are now populated on the GPU, but shaders don't read them yet.

---

### Task 4: Update `getTriangleGeometry()` and `__closesthit__triangle()` to use per-instance buffers in IAS mode

**Files:**
- Modify: `optix-jni/src/main/native/shaders/hit_triangle.cu:27-93` (getTriangleGeometry)
- Modify: `optix-jni/src/main/native/shaders/hit_triangle.cu:168-189` (__closesthit__triangle)

This is the core fix. We add an overloaded `getTriangleGeometry()` that accepts per-instance buffer pointers, and update `__closesthit__triangle()` to call it in IAS mode.

- [ ] **Step 4.1: Add the per-instance overload of `getTriangleGeometry()`**

In `optix-jni/src/main/native/shaders/hit_triangle.cu`, after the existing `getTriangleGeometry(const TriangleHitGroupData* hit_data)` function (line 93), add a new overload that takes explicit buffer pointers:

```cpp
/**
 * Get interpolated geometry from per-instance buffer pointers (IAS mode).
 *
 * In IAS mode with mixed triangle meshes, each instance has its own
 * vertex/index buffers stored in InstanceMaterial. This overload reads
 * from those per-instance pointers instead of the SBT hit group data.
 */
__device__ TriangleGeometry getTriangleGeometry(
    const float* vertices,
    const unsigned int* indices,
    unsigned int vertex_stride
) {
    TriangleGeometry geom;

    // Get hit point
    geom.t = optixGetRayTmax();
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    geom.hit_point = ray_origin + ray_direction * geom.t;

    // Get triangle primitive index and barycentric coordinates
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const float u = barycentrics.x;
    const float v = barycentrics.y;
    const float w = 1.0f - u - v;

    // Get vertex indices for this triangle
    const unsigned int idx0 = indices[prim_idx * 3 + 0];
    const unsigned int idx1 = indices[prim_idx * 3 + 1];
    const unsigned int idx2 = indices[prim_idx * 3 + 2];

    // Get vertex stride
    const unsigned int stride = vertex_stride;

    // Vertices are interleaved: [px, py, pz, nx, ny, nz, (u, v)] = stride floats per vertex
    const float* v0 = &vertices[idx0 * stride];
    const float* v1 = &vertices[idx1 * stride];
    const float* v2 = &vertices[idx2 * stride];

    // Interpolate normal using barycentric coordinates
    float3 normal = make_float3(
        w * v0[3] + u * v1[3] + v * v2[3],
        w * v0[4] + u * v1[4] + v * v2[4],
        w * v0[5] + u * v1[5] + v * v2[5]
    );
    normal = normalize(normal);

    // Interpolate UV coordinates if available (stride >= 8)
    geom.uv_coords = make_float2(0.0f, 0.0f);
    if (stride >= VERTEX_STRIDE_WITH_UV) {
        geom.uv_coords = make_float2(
            w * v0[6] + u * v1[6] + v * v2[6],
            w * v0[7] + u * v1[7] + v * v2[7]
        );
    }

    // Interpolate per-vertex alpha if available (stride >= 9)
    float vertex_alpha = 1.0f;
    if (stride >= VERTEX_STRIDE_WITH_ALPHA) {
        vertex_alpha = w * v0[8] + u * v1[8] + v * v2[8];
    }
    geom.vertex_alpha = vertex_alpha;

    // Determine if ray is entering or exiting (front face = entering)
    geom.entering = (dot(ray_direction, normal) < 0.0f);

    // Flip normal to face incoming ray
    geom.normal = geom.entering ? normal : make_float3(-normal.x, -normal.y, -normal.z);

    return geom;
}
```

- [ ] **Step 4.2: Update `__closesthit__triangle()` to branch on IAS mode**

In `__closesthit__triangle()` (line 168), replace the geometry and material retrieval section. Change lines 168-189 from:

```cpp
extern "C" __global__ void __closesthit__triangle() {
    // Get triangle hit group data from SBT
    const TriangleHitGroupData* hit_data = reinterpret_cast<TriangleHitGroupData*>(optixGetSbtDataPointer());

    // Get interpolated geometry (hit point, normal, UVs)
    const TriangleGeometry geom = getTriangleGeometry(hit_data);
    const float3 ray_direction = optixGetWorldRayDirection();

    // Get current depth from payload
    const unsigned int depth = optixGetPayload_3();

    // Track depth statistics
    if (params.stats) {
        atomicMax(&params.stats->max_depth_reached, depth + 1);
        atomicMin(&params.stats->min_depth_reached, depth + 1);
    }

    // Get material properties including PBR values (color, IOR, roughness, metallic, specular, film_thickness, emission)
    float4 mesh_color;
    float mesh_ior, roughness, metallic, specular, film_thickness, mesh_emission;
    getTriangleMaterial(hit_data, geom.uv_coords, hit_data->vertex_stride, geom.vertex_alpha,
                       mesh_color, mesh_ior, roughness, metallic, specular, film_thickness, mesh_emission);
```

To:

```cpp
extern "C" __global__ void __closesthit__triangle() {
    // Get triangle hit group data from SBT
    const TriangleHitGroupData* hit_data = reinterpret_cast<TriangleHitGroupData*>(optixGetSbtDataPointer());

    // Get interpolated geometry (hit point, normal, UVs)
    // In IAS mode, use per-instance buffer pointers from InstanceMaterial
    // In single-object mode, use SBT hit group data
    TriangleGeometry geom;
    unsigned int active_vertex_stride;
    if (params.use_ias && params.instance_materials) {
        const unsigned int instance_id = optixGetInstanceId();
        const InstanceMaterial& mat = params.instance_materials[instance_id];
        if (mat.vertices && mat.indices) {
            geom = getTriangleGeometry(mat.vertices, mat.indices, mat.vertex_stride);
            active_vertex_stride = mat.vertex_stride;
        } else {
            // Fallback for non-triangle instances that somehow hit this shader
            geom = getTriangleGeometry(hit_data);
            active_vertex_stride = hit_data->vertex_stride;
        }
    } else {
        geom = getTriangleGeometry(hit_data);
        active_vertex_stride = hit_data->vertex_stride;
    }
    const float3 ray_direction = optixGetWorldRayDirection();

    // Get current depth from payload
    const unsigned int depth = optixGetPayload_3();

    // Track depth statistics
    if (params.stats) {
        atomicMax(&params.stats->max_depth_reached, depth + 1);
        atomicMin(&params.stats->min_depth_reached, depth + 1);
    }

    // Get material properties including PBR values (color, IOR, roughness, metallic, specular, film_thickness, emission)
    float4 mesh_color;
    float mesh_ior, roughness, metallic, specular, film_thickness, mesh_emission;
    getTriangleMaterial(hit_data, geom.uv_coords, active_vertex_stride, geom.vertex_alpha,
                       mesh_color, mesh_ior, roughness, metallic, specular, film_thickness, mesh_emission);
```

**Important:** The `active_vertex_stride` variable replaces the previous `hit_data->vertex_stride` in the `getTriangleMaterial` call. This ensures the vertex stride matches the actual mesh being hit, not the SBT's default mesh.

- [ ] **Step 4.3: Also update the coverage blend vertex_stride check**

Further down in `__closesthit__triangle()` (around line 233), there's a check for vertex alpha:
```cpp
    const bool has_vertex_alpha_channel = hit_data->vertex_stride >= VERTEX_STRIDE_WITH_ALPHA;
```

Replace it with:
```cpp
    const bool has_vertex_alpha_channel = active_vertex_stride >= VERTEX_STRIDE_WITH_ALPHA;
```

- [ ] **Step 4.4: Compile to verify**

Run: `sbt "project optixJni" nativeCompile`
Expected: Compiles successfully. The shader now reads per-instance buffers in IAS mode.

- [ ] **Step 4.5: Run existing tests**

Run: `sbt test`
Expected: All tests pass. Single-object mode is unchanged (uses SBT path). Existing IAS tests with same-type meshes continue to work because their per-instance pointers point to the correct (same) mesh.

---

### Task 5: Update `__closesthit__photon()` to use per-instance buffers in IAS mode

**Files:**
- Modify: `optix-jni/src/main/native/shaders/caustics_ppm.cu:710-714` (photon closest hit, triangle branch)

- [ ] **Step 5.1: Update the triangle branch in `__closesthit__photon()`**

In `optix-jni/src/main/native/shaders/caustics_ppm.cu`, find `__closesthit__photon()` (line 697). The triangle branch at lines 710-714 currently reads:

```cpp
    if (optixGetPrimitiveType() == OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
        // Triangle mesh: interpolate normal from vertex data
        const TriangleHitGroupData* hit_data =
            reinterpret_cast<const TriangleHitGroupData*>(optixGetSbtDataPointer());
        const TriangleGeometry geom = getTriangleGeometry(hit_data);
```

Replace with:

```cpp
    if (optixGetPrimitiveType() == OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
        // Triangle mesh: interpolate normal from vertex data
        const TriangleHitGroupData* hit_data =
            reinterpret_cast<const TriangleHitGroupData*>(optixGetSbtDataPointer());
        // In IAS mode, use per-instance buffer pointers
        TriangleGeometry geom;
        if (params.use_ias && params.instance_materials) {
            const unsigned int id = optixGetInstanceId();
            const InstanceMaterial& mat = params.instance_materials[id];
            if (mat.vertices && mat.indices) {
                geom = getTriangleGeometry(mat.vertices, mat.indices, mat.vertex_stride);
            } else {
                geom = getTriangleGeometry(hit_data);
            }
        } else {
            geom = getTriangleGeometry(hit_data);
        }
```

**Note:** The `id` variable is already declared later (line 717) for IOR lookup. We must use a different variable name here or restructure to avoid the redeclaration. Since the existing code at line 717 declares `const unsigned int id = optixGetInstanceId();`, and our new code above also uses `id` in IAS mode, we need to unify these. The cleanest approach: rename the variable in our new block to `inst_id` and also use it for the IOR lookup below:

```cpp
    if (optixGetPrimitiveType() == OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
        // Triangle mesh: interpolate normal from vertex data
        const TriangleHitGroupData* hit_data =
            reinterpret_cast<const TriangleHitGroupData*>(optixGetSbtDataPointer());
        // In IAS mode, use per-instance buffer pointers
        TriangleGeometry geom;
        const unsigned int inst_id = optixGetInstanceId();
        if (params.use_ias && params.instance_materials) {
            const InstanceMaterial& mat = params.instance_materials[inst_id];
            if (mat.vertices && mat.indices) {
                geom = getTriangleGeometry(mat.vertices, mat.indices, mat.vertex_stride);
            } else {
                geom = getTriangleGeometry(hit_data);
            }
        } else {
            geom = getTriangleGeometry(hit_data);
        }
        outward_normal = geom.entering ? geom.normal : make_float3(-geom.normal.x, -geom.normal.y, -geom.normal.z);
        // IOR from instance material (IAS mode)
        ior_material = params.instance_materials[inst_id].ior;
        for (int i = 0; i < 4; i++) glass_color[i] = params.instance_materials[inst_id].color[i];
        glass_scale = 1.0f;
```

This replaces lines 710-720 of the original. The key change: `const unsigned int id` → `const unsigned int inst_id`, and the subsequent references at lines 717-719 use `inst_id` instead of `id`.

- [ ] **Step 5.2: Compile to verify**

Run: `sbt "project optixJni" nativeCompile`
Expected: Compiles successfully.

- [ ] **Step 5.3: Run all tests**

Run: `sbt test`
Expected: All tests pass.

---

### Task 6: Add end-to-end Scala tests for mixed mesh types

**Files:**
- Modify: `optix-jni/src/test/scala/menger/optix/InstanceAccelerationSuite.scala` (add new tests)
- Modify: `optix-jni/src/test/scala/menger/optix/TestUtilities.scala` (add mesh creation helpers)

- [ ] **Step 6.1: Add a `createSmallCubeMesh()` helper to TestUtilities**

In `optix-jni/src/test/scala/menger/optix/TestUtilities.scala`, after the existing `createScaledCubeMesh` method (around line 267), add a helper that creates a mesh with a DIFFERENT vertex count than the unit cube — this is critical for the test because same-sized meshes wouldn't trigger the out-of-bounds read:

```scala
  /** Create a large subdivided cube mesh for mixed-mesh testing.
    * Has many more vertices than createUnitCubeMesh (24), ensuring
    * that primitive IDs from this mesh would be out-of-bounds if
    * indexed into the unit cube's vertex buffer.
    */
  def createLargeSubdividedCubeMesh(): menger.common.TriangleMeshData =
    createSubdividedCubeMesh(5)
```

- [ ] **Step 6.2: Add mixed-mesh rendering tests to InstanceAccelerationSuite**

In `optix-jni/src/test/scala/menger/optix/InstanceAccelerationSuite.scala`, after the "Sequential setTriangleMesh" section (after line 269), add a new test section:

```scala
  // ================================================
  // Mixed Triangle Mesh Type Tests (per-instance buffers)
  // ================================================

  "Mixed triangle mesh types" should
    "render two instances from different meshes without CUDA error" taggedAs (Slow) in:
      // Mesh A: unit cube (24 vertices, 12 triangles)
      val meshA = TestUtilities.createUnitCubeMesh()
      renderer.setTriangleMesh(meshA)
      renderer.addTriangleMeshInstance(
        Vector[3](-1.5f, 0.0f, 0.0f), OPAQUE_RED, 1.5f
      )

      // Mesh B: large subdivided cube (much more vertices/triangles)
      // If per-instance buffers don't work, primitive IDs from mesh B
      // will index into mesh A's smaller buffer -> CUDA error 700
      val meshB = TestUtilities.createLargeSubdividedCubeMesh()
      renderer.setTriangleMesh(meshB)
      renderer.addTriangleMeshInstance(
        Vector[3](1.5f, 0.0f, 0.0f), OPAQUE_BLUE, 1.5f
      )

      // This render previously crashed with CUDA error 700
      val img = renderImage(TEST_IMAGE_SIZE)
      img.length shouldBe ImageValidation.imageByteSize(TEST_IMAGE_SIZE)
      img.count(_ != 0) should be > 0

  it should
    "render three instances from two different meshes" taggedAs (Slow) in:
      // Two instances from mesh A
      val meshA = TestUtilities.createUnitCubeMesh()
      renderer.setTriangleMesh(meshA)
      renderer.addTriangleMeshInstance(
        Vector[3](-2.0f, 0.0f, 0.0f), OPAQUE_RED, 1.5f
      )
      renderer.addTriangleMeshInstance(
        Vector[3](0.0f, 0.0f, 0.0f), OPAQUE_GREEN, 1.5f
      )

      // One instance from mesh B
      val meshB = TestUtilities.createLargeSubdividedCubeMesh()
      renderer.setTriangleMesh(meshB)
      renderer.addTriangleMeshInstance(
        Vector[3](2.0f, 0.0f, 0.0f), OPAQUE_BLUE, 1.5f
      )

      val img = renderImage(TEST_IMAGE_SIZE)
      img.length shouldBe ImageValidation.imageByteSize(TEST_IMAGE_SIZE)
      img.count(_ != 0) should be > 0

  it should
    "render mixed meshes with sphere instances" taggedAs (Slow) in:
      // Sphere instance
      renderer.addSphereInstance(
        Vector[3](-2.0f, 0.0f, 0.0f), OPAQUE_RED, 1.5f
      )

      // Triangle mesh A
      val meshA = TestUtilities.createUnitCubeMesh()
      renderer.setTriangleMesh(meshA)
      renderer.addTriangleMeshInstance(
        Vector[3](0.0f, 0.0f, 0.0f), OPAQUE_GREEN, 1.5f
      )

      // Triangle mesh B (different size)
      val meshB = TestUtilities.createLargeSubdividedCubeMesh()
      renderer.setTriangleMesh(meshB)
      renderer.addTriangleMeshInstance(
        Vector[3](2.0f, 0.0f, 0.0f), OPAQUE_BLUE, 1.5f
      )

      val img = renderImage(TEST_IMAGE_SIZE)
      img.length shouldBe ImageValidation.imageByteSize(TEST_IMAGE_SIZE)
      img.count(_ != 0) should be > 0
```

- [ ] **Step 6.3: Run only the new tests to verify they fail BEFORE the shader fix is in place (if Tasks 4-5 not yet done) or pass (if Tasks 4-5 are done)**

Run: `sbt "testOnly menger.optix.InstanceAccelerationSuite -- -z \"Mixed triangle mesh types\""`

If Tasks 1-5 are already implemented: Expected: All 3 tests PASS.
If Tasks 1-3 done but Tasks 4-5 not yet done: Expected: CUDA error 700 (the tests correctly catch the bug the shader fix addresses).

- [ ] **Step 6.4: Run full test suite**

Run: `sbt test`
Expected: All tests pass (27 C++ + all Scala tests).

---

## Execution Order

Tasks 1-3 are **data plumbing** (store mesh_index, add fields, populate in buildIAS). They can be done in sequence and verified with compilation only.

Tasks 4-5 are the **shader fix** (read per-instance data). They depend on Tasks 1-3.

Task 6 is the **test** that proves the fix works end-to-end. Ideally written before Tasks 4-5 (TDD: write failing test first), but since the test requires GPU rendering and would crash with CUDA error 700, writing it after the fix is also acceptable.

**Recommended execution:** Tasks 1 → 2 → 3 → 4 → 5 → 6 (sequential, each building on prior).

**Commit strategy:** Single commit after all 6 tasks pass, since they form one atomic fix.
