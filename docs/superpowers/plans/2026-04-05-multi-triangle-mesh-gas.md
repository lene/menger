# Multi-Triangle-Mesh GAS Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support multiple simultaneous triangle meshes in OptiXWrapper, each with independent GPU buffers and GAS handles, so that scenes with mixed sponge types/levels render correctly.

**Architecture:** Replace the single `TriangleMeshGPU` struct in `OptiXWrapper::Impl` with a `std::vector<TriangleMeshGPU>`. Each call to `setTriangleMesh()` pushes a new entry. `addTriangleMeshInstance()` references the latest mesh entry's GAS. `clearAllInstances()` and `clearTriangleMesh()` free all entries. The cylinder pattern (each cylinder gets its own GAS in `cylinder_gas_buffers`) is the existing precedent.

**Tech Stack:** C++17, CUDA, OptiX 9.0, Scala 3 (test layer)

**Key Insight:** The `buildIAS()` method already uses per-instance `inst.gas_handle` (line 464), so it works correctly as long as each instance stores the right GAS handle. The sphere uses a shared GAS via `gas_registry[GEOMETRY_TYPE_SPHERE]`. Cylinders each get a unique GAS. Triangle meshes currently share one global slot — we need to follow the cylinder pattern instead.

---

### Task 1: Replace single TriangleMeshGPU with vector

**Files:**
- Modify: `optix-jni/src/main/native/OptiXWrapper.cpp:42-48` (Impl struct)
- Modify: `optix-jni/src/main/native/OptiXWrapper.cpp:50-52` (mesh_aabb fields)

- [ ] **Step 1.1: Change the single `triangle_mesh_gpu` to a vector**

In `OptiXWrapper.cpp`, replace the `Impl` struct's single `TriangleMeshGPU triangle_mesh_gpu;` with a vector, and add a helper to get the current (latest) mesh entry:

```cpp
// Triangle mesh GPU state — one entry per distinct mesh uploaded via setTriangleMesh()
struct TriangleMeshGPU {
    CUdeviceptr d_vertices = 0;           // GPU vertex buffer
    CUdeviceptr d_indices = 0;            // GPU index buffer
    OptixTraversableHandle gas_handle = 0; // Triangle GAS
    CUdeviceptr d_gas_output_buffer = 0;  // GAS memory
    bool gas_built = false;               // True if GAS is ready
    unsigned int num_vertices = 0;        // Vertex count for this mesh
    unsigned int num_triangles = 0;       // Triangle count for this mesh
    unsigned int vertex_stride = 8;       // Floats per vertex (default: pos+normal+uv)
};
std::vector<TriangleMeshGPU> triangle_meshes;
```

Note: `num_vertices`, `num_triangles`, and `vertex_stride` are stored per-mesh because `buildTriangleMeshGAS()` needs them, and `SceneParameters` only stores the latest mesh's metadata.

- [ ] **Step 1.2: Compile to verify struct change**

Run: `sbt "project optixJni" nativeCompile`
Expected: Compilation errors in `setTriangleMesh`, `buildTriangleMeshGAS`, `addTriangleMeshInstance`, `clearTriangleMesh`, `clearAllInstances`, `dispose` — all referencing the removed `triangle_mesh_gpu` field. This confirms all call sites that need updating.

---

### Task 2: Update `setTriangleMesh()` to push new mesh entries

**Files:**
- Modify: `optix-jni/src/main/native/OptiXWrapper.cpp:263-321` (setTriangleMesh)

- [ ] **Step 2.1: Rewrite setTriangleMesh to append a new mesh entry**

Replace the current `setTriangleMesh` body. Instead of freeing the old buffers and overwriting the single slot, push a new `TriangleMeshGPU` entry:

```cpp
void OptiXWrapper::setTriangleMesh(
    const float* vertices,
    unsigned int num_vertices,
    const unsigned int* indices,
    unsigned int num_triangles,
    unsigned int vertex_stride
) {
    Impl::TriangleMeshGPU mesh_entry;

    // Allocate and copy vertex buffer (vertex_stride floats per vertex)
    size_t vertex_size = num_vertices * vertex_stride * sizeof(float);
    cudaMalloc(reinterpret_cast<void**>(&mesh_entry.d_vertices), vertex_size);
    cudaMemcpy(
        reinterpret_cast<void*>(mesh_entry.d_vertices),
        vertices,
        vertex_size,
        cudaMemcpyHostToDevice
    );

    // Allocate and copy index buffer (3 indices per triangle)
    size_t index_size = num_triangles * 3 * sizeof(unsigned int);
    cudaMalloc(reinterpret_cast<void**>(&mesh_entry.d_indices), index_size);
    cudaMemcpy(
        reinterpret_cast<void*>(mesh_entry.d_indices),
        indices,
        index_size,
        cudaMemcpyHostToDevice
    );

    // Store mesh metadata for GAS building
    mesh_entry.num_vertices = num_vertices;
    mesh_entry.num_triangles = num_triangles;
    mesh_entry.vertex_stride = vertex_stride;
    mesh_entry.gas_built = false;

    impl->triangle_meshes.push_back(mesh_entry);

    // Compute mesh AABB from vertex positions (for caustic target)
    impl->mesh_aabb_min = {FLT_MAX, FLT_MAX, FLT_MAX};
    impl->mesh_aabb_max = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (unsigned int i = 0; i < num_vertices; ++i) {
        const float* v = vertices + i * vertex_stride;
        impl->mesh_aabb_min.x = fminf(impl->mesh_aabb_min.x, v[0]);
        impl->mesh_aabb_min.y = fminf(impl->mesh_aabb_min.y, v[1]);
        impl->mesh_aabb_min.z = fminf(impl->mesh_aabb_min.z, v[2]);
        impl->mesh_aabb_max.x = fmaxf(impl->mesh_aabb_max.x, v[0]);
        impl->mesh_aabb_max.y = fmaxf(impl->mesh_aabb_max.y, v[1]);
        impl->mesh_aabb_max.z = fmaxf(impl->mesh_aabb_max.z, v[2]);
    }

    // Update scene parameters (for SBT setup — uses latest mesh metadata)
    impl->scene.setTriangleMeshMeta(num_vertices, num_triangles);
    auto& mesh_params = impl->scene.getTriangleMeshMutable();
    mesh_params.d_vertices = mesh_entry.d_vertices;
    mesh_params.d_indices = mesh_entry.d_indices;
    mesh_params.vertex_stride = vertex_stride;
}
```

- [ ] **Step 2.2: Compile to verify**

Run: `sbt "project optixJni" nativeCompile`
Expected: Still errors in other functions referencing `triangle_mesh_gpu`, but `setTriangleMesh` should compile.

---

### Task 3: Update `buildTriangleMeshGAS()` to work with a specific mesh index

**Files:**
- Modify: `optix-jni/src/main/native/OptiXWrapper.cpp:385-420` (buildTriangleMeshGAS)
- Modify: `optix-jni/src/main/native/include/OptiXWrapper.h:137` (declaration)

- [ ] **Step 3.1: Add index parameter to buildTriangleMeshGAS**

In `OptiXWrapper.h`, change the declaration:

```cpp
void buildTriangleMeshGAS(size_t mesh_index);  // Build GAS for specific triangle mesh
```

In `OptiXWrapper.cpp`, rewrite the function:

```cpp
void OptiXWrapper::buildTriangleMeshGAS(size_t mesh_index) {
    if (mesh_index >= impl->triangle_meshes.size()) {
        std::cerr << "[OptiX] buildTriangleMeshGAS: invalid mesh index "
                  << mesh_index << " (have " << impl->triangle_meshes.size() << " meshes)"
                  << std::endl;
        return;
    }

    auto& mesh = impl->triangle_meshes[mesh_index];

    // Free existing GAS if any
    if (mesh.d_gas_output_buffer) {
        cudaFree(reinterpret_cast<void*>(mesh.d_gas_output_buffer));
        mesh.d_gas_output_buffer = 0;
    }

    if (mesh.num_triangles == 0) {
        mesh.gas_handle = 0;
        mesh.gas_built = false;
        return;
    }

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags =
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptiXContext::GASBuildResult result = impl->optix_context.buildTriangleGAS(
        mesh.d_vertices,
        mesh.num_vertices,
        mesh.d_indices,
        mesh.num_triangles,
        accel_options,
        mesh.vertex_stride
    );

    mesh.d_gas_output_buffer = result.gas_buffer;
    mesh.gas_handle = result.handle;
    mesh.gas_built = true;
}
```

- [ ] **Step 3.2: Compile to verify**

Run: `sbt "project optixJni" nativeCompile`
Expected: Errors at the `buildTriangleMeshGAS()` call sites (in `addTriangleMeshInstance` and the old no-arg call if any) — those are fixed in the next task.

---

### Task 4: Update `addTriangleMeshInstance()` to reference the latest mesh

**Files:**
- Modify: `optix-jni/src/main/native/OptiXWrapper.cpp:957-1023` (addTriangleMeshInstance)

- [ ] **Step 4.1: Rewrite addTriangleMeshInstance to use mesh vector**

Replace the body of `addTriangleMeshInstance`:

```cpp
int OptiXWrapper::addTriangleMeshInstance(
    const float* transform, float r, float g, float b, float a, float ior,
    float roughness, float metallic, float specular, float emission, int textureIndex,
    float film_thickness
) {
    if (impl->instances.size() >= impl->max_instances) {
        if (!impl->max_instances_warning_shown) {
            std::cerr << "[OptiX][TriangleMesh] Maximum instances ("
                      << impl->max_instances << ") reached" << std::endl;
            impl->max_instances_warning_shown = true;
        }
        return -1;
    }

    // Check if any mesh data exists
    if (impl->triangle_meshes.empty()) {
        std::cerr << "[OptiX] Cannot add triangle mesh instance: no mesh set"
                  << " (call setTriangleMesh first)" << std::endl;
        return -1;
    }

    // Use the latest mesh entry
    size_t mesh_index = impl->triangle_meshes.size() - 1;
    auto& mesh = impl->triangle_meshes[mesh_index];

    // Auto-build triangle GAS if not built yet
    if (!mesh.gas_built) {
        buildTriangleMeshGAS(mesh_index);
    }

    Impl::ObjectInstance inst;
    inst.geometry_type = GEOMETRY_TYPE_TRIANGLE;
    inst.gas_handle = mesh.gas_handle;
    std::memcpy(inst.transform, transform, 12 * sizeof(float));
    inst.color[0] = r;
    inst.color[1] = g;
    inst.color[2] = b;
    inst.color[3] = a;
    inst.ior = ior;
    inst.roughness = roughness;
    inst.metallic = metallic;
    inst.specular = specular;
    inst.emission = emission;
    inst.film_thickness = film_thickness;
    inst.texture_index = textureIndex;
    inst.active = true;

    int instanceId = static_cast<int>(impl->instances.size());
    impl->instances.push_back(inst);
    impl->ias_dirty = true;

    // Force pipeline rebuild when entering IAS mode for first time
    if (!impl->use_ias) {
        impl->pipeline_built = false;
    }
    impl->use_ias = true;

    return instanceId;
}
```

Key changes from the old code:
- Reads from `triangle_meshes.back()` instead of the single `triangle_mesh_gpu`
- Stores per-mesh GAS handle directly in the instance (no `gas_registry` for triangles)
- Calls `buildTriangleMeshGAS(mesh_index)` with the specific index

- [ ] **Step 4.2: Compile to verify**

Run: `sbt "project optixJni" nativeCompile`
Expected: Errors only in `clearTriangleMesh`, `clearAllInstances`, `hasTriangleMesh`, and `dispose`.

---

### Task 5: Update `clearTriangleMesh()` and `clearAllInstances()`

**Files:**
- Modify: `optix-jni/src/main/native/OptiXWrapper.cpp:331-350` (clearTriangleMesh)
- Modify: `optix-jni/src/main/native/OptiXWrapper.cpp:1162-1218` (clearAllInstances)
- Modify: `optix-jni/src/main/native/OptiXWrapper.cpp:1320-1364` (dispose)

- [ ] **Step 5.1: Add a helper to free all triangle mesh GPU entries**

Add this private helper method after `clearTriangleMesh()` (around line 350). Also add its declaration in the private section of `OptiXWrapper.h` if needed (it's an internal helper, so it can just be a standalone function or added to Impl):

Actually, since this is used from multiple places, implement it as a local lambda or just inline it. The simplest approach: add a `freeAllTriangleMeshes()` helper in the Impl struct or as a free function inside the .cpp file.

Add this function before `clearTriangleMesh`:

```cpp
// Free all triangle mesh GPU resources
static void freeTriangleMeshes(std::vector<OptiXWrapper::Impl::TriangleMeshGPU>& meshes) {
    for (auto& mesh : meshes) {
        if (mesh.d_vertices) {
            cudaFree(reinterpret_cast<void*>(mesh.d_vertices));
        }
        if (mesh.d_indices) {
            cudaFree(reinterpret_cast<void*>(mesh.d_indices));
        }
        if (mesh.d_gas_output_buffer) {
            cudaFree(reinterpret_cast<void*>(mesh.d_gas_output_buffer));
        }
    }
    meshes.clear();
}
```

Note: `Impl` is a private struct, so the static function needs to be inside the `.cpp` file after the struct definition. Since `Impl::TriangleMeshGPU` is a nested type, the function signature needs to reference it. An alternative is to make it a method on Impl. The simplest approach: just put the cleanup code in each call site rather than a shared helper, since there are only 3 call sites and the code is straightforward. Let's inline it to avoid access issues.

- [ ] **Step 5.2: Rewrite clearTriangleMesh**

```cpp
void OptiXWrapper::clearTriangleMesh() {
    // Free all triangle mesh GPU buffers
    for (auto& mesh : impl->triangle_meshes) {
        if (mesh.d_vertices) {
            cudaFree(reinterpret_cast<void*>(mesh.d_vertices));
        }
        if (mesh.d_indices) {
            cudaFree(reinterpret_cast<void*>(mesh.d_indices));
        }
        if (mesh.d_gas_output_buffer) {
            cudaFree(reinterpret_cast<void*>(mesh.d_gas_output_buffer));
        }
    }
    impl->triangle_meshes.clear();

    impl->scene.clearTriangleMesh();
}
```

- [ ] **Step 5.3: Update clearAllInstances**

In `clearAllInstances`, after clearing cylinder GAS buffers (line 1206), add triangle mesh cleanup. The triangle mesh GAS buffers are now per-mesh and NOT in `gas_registry`, so the gas_registry only holds sphere GAS. But we should still clear the meshes on `clearAllInstances`:

After the line `impl->cylinder_gas_buffers.clear();` (line 1206), add:

```cpp
    // Free triangle mesh GAS buffers (each mesh has its own GAS)
    for (auto& mesh : impl->triangle_meshes) {
        if (mesh.d_vertices) {
            cudaFree(reinterpret_cast<void*>(mesh.d_vertices));
        }
        if (mesh.d_indices) {
            cudaFree(reinterpret_cast<void*>(mesh.d_indices));
        }
        if (mesh.d_gas_output_buffer) {
            cudaFree(reinterpret_cast<void*>(mesh.d_gas_output_buffer));
        }
    }
    impl->triangle_meshes.clear();
```

- [ ] **Step 5.4: Verify dispose is safe**

In `dispose()` (line 1320+), `clearAllInstances()` is called first, which now frees all triangle mesh GPU resources and clears `impl->triangle_meshes`. The subsequent `gas_registry` loop only needs to handle sphere GAS (which is still stored there). No additional triangle mesh cleanup is needed in `dispose()` — it's already handled by `clearAllInstances()`.

Verify: read through `dispose()` and confirm `clearAllInstances()` is called before the `gas_registry` cleanup, and that no code after it references `triangle_meshes`.

- [ ] **Step 5.5: Update hasTriangleMesh**

The `hasTriangleMesh()` method at line 352 currently delegates to `impl->scene.hasTriangleMesh()`. This should still work since `setTriangleMesh` still calls `scene.setTriangleMeshMeta()`. No change needed here.

- [ ] **Step 5.6: Compile and run C++ tests**

Run: `sbt "project optixJni" nativeCompile ; sbt "project optixJni" nativeTest`
Expected: Compilation succeeds. All 27 C++ Google Tests pass.

---

### Task 6: Remove triangle mesh from gas_registry

**Files:**
- Modify: `optix-jni/src/main/native/OptiXWrapper.cpp:986-992` (was in addTriangleMeshInstance, already removed in Task 4)

- [ ] **Step 6.1: Verify gas_registry no longer referenced for triangles**

The old code at lines 986-992 wrote `gas_registry[GEOMETRY_TYPE_TRIANGLE]`. Task 4 already removed this. Verify by searching the file for `GEOMETRY_TYPE_TRIANGLE` — it should only appear in the `inst.geometry_type = GEOMETRY_TYPE_TRIANGLE` assignment.

Run: `grep GEOMETRY_TYPE_TRIANGLE optix-jni/src/main/native/OptiXWrapper.cpp`
Expected: Only the `inst.geometry_type = GEOMETRY_TYPE_TRIANGLE;` line.

- [ ] **Step 6.2: Compile to verify**

Run: `sbt "project optixJni" nativeCompile`
Expected: Clean compilation.

---

### Task 7: Run full test suite

**Files:**
- Test: `optix-jni/src/test/scala/menger/optix/InstanceAccelerationSuite.scala`

- [ ] **Step 7.1: Run all Scala tests**

Run: `sbt test`
Expected: All ~1,080 tests pass, including the 28 InstanceAccelerationSuite tests. The existing "Sequential setTriangleMesh" tests (lines 224-269) should continue to pass — they test the exact scenario we're fixing.

- [ ] **Step 7.2: Run end-to-end test 55 (3D Mixed frac levels)**

Run:
```bash
__GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a menger-app/target/universal/stage/bin/menger-app \
  -o --objects "type=sponge-surface:level=1.5" --objects "type=sponge-surface:level=2.5:x=3" \
  --timeout 3 -s test55_multimesh.png
```
Expected: Image shows TWO distinct sponges (not just one). The first sponge (level 1.5) should be visible at the default position, and the second (level 2.5) should be visible offset to the right.

- [ ] **Step 7.3: Run end-to-end test 56 (3D Volume vs Surface frac)**

Run:
```bash
__GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a menger-app/target/universal/stage/bin/menger-app \
  -o --objects "type=sponge-volume:level=1.5" --objects "type=sponge-surface:level=1.5:x=3" \
  --timeout 3 -s test56_multimesh.png
```
Expected: Image shows TWO distinct sponges — one volume, one surface — without CUDA error 700.

- [ ] **Step 7.4: Run end-to-end test 41 (Mixed 4D sponges)**

Run:
```bash
__GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a menger-app/target/universal/stage/bin/menger-app \
  -o --objects "type=tesseract-sponge:level=1:color=red" \
  --objects "type=tesseract-sponge:level=1:x=3:color=green" \
  --timeout 3 -s test41_multimesh.png
```
Expected: Image shows TWO colored tesseract sponges (red and green), both visible.

---

### Task 8: Commit

**Files:**
- All modified files from Tasks 1-6
- Include the existing uncommitted changes (TriangleMeshSceneBuilder.scala compatibility relaxation, FractionalLevelSceneBuilderSuite.scala, TestUtilities.scala)

- [ ] **Step 8.1: Run sbt test one final time**

Run: `sbt test`
Expected: All tests pass.

- [ ] **Step 8.2: Show diff to user for review**

Run: `git diff`
Show output to user. Wait for user approval.

- [ ] **Step 8.3: Commit (after user approval)**

```bash
git add optix-jni/src/main/native/OptiXWrapper.cpp
git add optix-jni/src/main/native/include/OptiXWrapper.h
git add menger-app/src/main/scala/menger/engines/scene/TriangleMeshSceneBuilder.scala
git add menger-app/src/test/scala/menger/engines/scene/FractionalLevelSceneBuilderSuite.scala
git add optix-jni/src/test/scala/menger/optix/InstanceAccelerationSuite.scala
git add optix-jni/src/test/scala/menger/optix/TestUtilities.scala
git commit -m "fix: Support multiple simultaneous triangle meshes in OptiXWrapper

Replace single TriangleMeshGPU slot with a vector so each setTriangleMesh()
call preserves previous mesh data. Each addTriangleMeshInstance() references
its specific mesh's GAS handle, preventing dangling handles when scenes
contain objects with different geometries (e.g., mixed sponge levels).

Root cause: setTriangleMesh() freed previous mesh buffers and rebuilt GAS,
invalidating earlier instances' GAS handles. Fixes manual tests 41, 55, 56.

Also includes:
- Relaxed TriangleMeshSceneBuilder.isCompatible for mixed 3D sponge types/levels
- Tests for multi-mesh compatibility validation"
```
