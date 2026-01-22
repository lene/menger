# Tesseract Edge Rendering Crash After Rotation

## Problem Description

Manual test #23 (Tesseract with chrome edges) crashes after rotating the view for 1-2 seconds in interactive mode.

**Command:**
```bash
./menger-app/target/universal/stage/bin/menger-app -o --objects type=tesseract:edge-material=chrome:edge-radius=0.02
```

**Behavior:**
- Initial render works fine (headless and interactive)
- Interactive mode starts successfully
- After rotating with mouse drag for 1-2 seconds, the application crashes
- Crash occurs during pipeline recreation, NOT during rendering

## Error Messages

```
[OptiX] Render failed: CUDA call 'cudaDeviceSynchronize()' failed: an illegal memory access was encountered (700)
[OptiX][2][PIPELINE]: Error synching on OptixPipeline event (CUDA error string: an illegal memory access was encountered, CUDA error code: 700)
[OptiX][2][COMPILER]: COMPILE ERROR: failed to create pipeline
Info: Pipeline statistics
        module(s)                            :     1
        entry function(s)                    :    13-16
        trace call(s)                        :    36
        continuation callable call(s)        :     0
        direct callable call(s)              :     0
        basic block(s) in entry functions    :   442-457
        instruction(s) in entry functions    :  6500-6563
        non-entry function(s)                :    32
        basic block(s) in non-entry functions:   336
        instruction(s) in non-entry functions:  5566
        debug information                    :    no

[OptiX] Render failed: OptiX call 'optixPipelineCreate(...)' failed: OPTIX_ERROR_PIPELINE_LINK_ERROR (7251)
```

**Key Observations:**
- Error occurs during `optixPipelineCreate()` call, not during ray tracing
- CUDA illegal memory access (700) happens during `cudaDeviceSynchronize()`
- Pipeline statistics show 13-16 entry functions (varies based on attempted fixes)
- Error repeats multiple times as the system tries to recreate the pipeline

## Root Cause Analysis

### When Does the Crash Occur?

The crash happens when:
1. User rotates the tesseract in interactive mode
2. OptiXEngine detects rotation event for 4D object (hasTesseracts = true)
3. `rebuildScene()` is called to regenerate the tesseract with new rotation
4. Scene rebuild attempts to recreate the OptiX pipeline
5. During `optixPipelineCreate()`, CUDA reports illegal memory access

### Why Rotation Triggers Rebuild

Tesseract edges are rendered as cylinders. When the 4D tesseract is rotated:
- The 4D→3D projection changes
- Edge positions (cylinder endpoints p0, p1) must be recalculated
- The entire scene must be rebuilt with new cylinder instances
- This triggers dispose/initialize or clearAllInstances cycle

### Technical Context

**Cylinder Edge Rendering:**
- Each tesseract edge is rendered as a custom primitive cylinder
- Cylinders use custom intersection shaders (`hit_cylinder.cu`)
- Each cylinder gets its own GAS (Geometry Acceleration Structure)
- Cylinder data is stored in `params.cylinder_data` array
- Instance materials reference cylinders via `texture_index` field (repurposed)

**Pipeline Configuration:**
- 3 geometry types: SPHERE (0), TRIANGLE (1), CYLINDER (2)
- 2 ray types per geometry: primary and shadow
- SBT has 6 records: [0-1]=sphere, [2-3]=triangle, [4-5]=cylinder
- SBT offset calculated as: `geometry_type * 2`

## Attempted Solutions

### 1. Fix Cylinder Shadow Hitgroup Program ❌

**Change:** `PipelineManager.cpp:107-114`
```cpp
// Changed from using sphere's __closesthit__shadow to cylinder-specific
cylinder_shadow_hitgroup_prog_group = optix_context.createHitgroupProgramGroup(
    cylinder_module, "__closesthit__cylinder_shadow",
    cylinder_module, "__intersection__cylinder"
);
```

**Rationale:** Cylinder shadows were using wrong closest hit program

**Result:** Still crashes after rotation

### 2. Add Anyhit Program Support ❌

**Changes:**
- `OptiXContext.h:62-70` - Added createHitgroupProgramGroup overload with anyhit
- `OptiXContext.cpp:254-286` - Implemented anyhit support
- `PipelineManager.cpp:105-114` - Registered anyhit programs for cylinders

**Rationale:** Cylinder shaders define anyhit programs for transparency, but they weren't being registered

**Result:** Still crashes, then REVERTED this change to isolate the issue

### 3. Add Bounds Checking to Cylinder Shader ❌

**Change:** `hit_cylinder.cu:24-35`
```cpp
// Added checks for:
// - instanceId < params.num_instances
// - cylinder_index >= 0 && < params.num_cylinders
// - Null pointer checks for params buffers
```

**Rationale:** Prevent out-of-bounds memory access in intersection shader

**Result:** Still crashes, then REVERTED - bounds checks access params during pipeline compilation which may not be initialized yet

### 4. Track and Free Cylinder GAS Buffers ❌

**Changes:** `OptiXWrapper.cpp`
```cpp
// Added cylinder_gas_buffers vector to track GAS for each cylinder
std::vector<GASData> cylinder_gas_buffers;

// Store GAS in tracking vector when creating cylinder
impl->cylinder_gas_buffers.push_back(gas_data);

// Free all cylinder GAS buffers in clearAllInstances()
for (const auto& gas : impl->cylinder_gas_buffers) {
    if (gas.gas_buffer) cudaFree(...);
    if (gas.aabb_buffer) cudaFree(...);
}
```

**Rationale:** Cylinders were creating GAS buffers that weren't being freed, causing memory leaks and stale references

**Result:** Memory leak fixed, but still crashes after rotation

### 5. Add CUDA Synchronization in Dispose ❌

**Change:** `OptiXWrapper.cpp:1103-1145`
```cpp
// Added cudaDeviceSynchronize() at end of dispose()
// Clear any pending CUDA errors
```

**Rationale:** Clear pending CUDA errors after cleanup

**Result:** Still crashes

### 6. Add Validation Before Launch ❌

**Change:** `OptiXWrapper.cpp:555-562`
```cpp
if (impl->use_ias) {
    if (impl->ias_handle == 0) {
        throw std::runtime_error("[OptiX] IAS handle is null but use_ias is true");
    }
    if (!impl->d_instance_materials) {
        throw std::runtime_error("[OptiX] Instance materials not uploaded to GPU");
    }
    // ...
}
```

**Rationale:** Catch invalid state before attempting to render

**Result:** No exceptions thrown, still crashes during pipeline creation

### 7. Avoid Destroying OptiX Context on Rebuild ❌ (CURRENT)

**Change:** `OptiXEngine.scala:231-275`
```scala
// BEFORE:
renderer.dispose()           // Destroys entire OptiX context
renderer.initialize(...)     // Recreates everything
// Reconfigure lights, plane, camera, etc.
builder.buildScene(...)      // Rebuild geometry

// AFTER:
renderer.clearAllInstances() // Only clears geometry instances
builder.buildScene(...)      // Rebuild geometry
// Restore camera state
```

**Rationale:** Destroying and recreating the OptiX context is too heavy-handed. Keep context alive and only rebuild geometry.

**Result:** Still crashes after rotation

## Current Code State

### Modified Files

1. **optix-jni/src/main/native/PipelineManager.cpp**
   - Line 105-114: Cylinder hitgroups use cylinder-specific programs (no anyhit)
   - Anyhit support code added but not currently used

2. **optix-jni/src/main/native/include/OptiXContext.h**
   - Line 62-70: Added createHitgroupProgramGroup overload with anyhit support

3. **optix-jni/src/main/native/OptiXContext.cpp**
   - Line 254-286: Implemented anyhit program support

4. **optix-jni/src/main/native/OptiXWrapper.cpp**
   - Line 88-89: Added cylinder_gas_buffers tracking vector
   - Line 925-931: Store cylinder GAS in tracking vector
   - Line 998-1008: Free cylinder GAS buffers in clearAllInstances()
   - Line 555-562: Added validation for IAS mode
   - Line 1132-1139: Added CUDA synchronization in dispose()

5. **optix-jni/src/main/native/shaders/hit_cylinder.cu**
   - No bounds checking (reverted)

6. **menger-app/src/main/scala/menger/engines/OptiXEngine.scala**
   - Line 231-275: Modified rebuildScene() to avoid dispose/initialize cycle

## Diagnostic Information Needed

### What We Know

1. ✅ Initial render works (both headless and interactive)
2. ✅ Static rendering works fine
3. ✅ Error occurs during pipeline creation, not rendering
4. ✅ Error is triggered by rotation events
5. ✅ Crash is specific to cylinder geometry (tesseract edges)
6. ✅ Other tests without cylinders work fine

### What We Don't Know

1. ❓ Why does pipeline creation fail after clearAllInstances()?
2. ❓ What memory is being accessed illegally during cudaDeviceSynchronize()?
3. ❓ Is the issue in shader code, SBT setup, or GAS configuration?
4. ❓ Does the issue occur during module creation, program group creation, or pipeline linking?
5. ❓ Are there stale CUDA memory references somewhere?

## Next Steps to Investigate

### 1. Add Detailed Logging

Add logging to track exactly where the crash occurs:

```cpp
// In PipelineManager::buildPipeline()
std::cerr << "[Debug] Starting pipeline rebuild" << std::endl;
std::cerr << "[Debug] Loading PTX modules" << std::endl;
loadPTXModules();
std::cerr << "[Debug] Creating program groups" << std::endl;
createProgramGroups();
std::cerr << "[Debug] Creating pipeline" << std::endl;
createPipeline();  // Crash likely happens here
std::cerr << "[Debug] Pipeline created successfully" << std::endl;
```

### 2. Check CUDA Memory State

Before pipeline creation, verify all CUDA allocations are valid:

```cpp
// After clearAllInstances(), verify:
// - impl->d_instance_materials is freed
// - impl->d_instances_buffer is freed
// - impl->d_cylinder_data is freed
// - All cylinder GAS buffers are freed
// - CUDA error state is clear
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    std::cerr << "[Debug] CUDA error before pipeline: " << cudaGetErrorString(err) << std::endl;
}
```

### 3. Test Without Cylinder Module

Try building pipeline without cylinder module to isolate whether issue is cylinder-specific:

```cpp
// In PipelineManager::loadPTXModules()
// Temporarily skip loading cylinder_module
// cylinder_module = nullptr;
```

### 4. Use cuda-memcheck

Run with CUDA memory checker to identify illegal access:

```bash
cuda-memcheck ./menger-app/target/universal/stage/bin/menger-app -o --objects type=tesseract:edge-material=chrome:edge-radius=0.02
```

### 5. Check for Module/Program Group Lifecycle Issues

Verify that:
- Modules aren't being destroyed while program groups reference them
- Program groups aren't being destroyed while pipeline references them
- No double-free or use-after-free in cleanup code

### 6. Try Alternative Rebuild Strategy

Instead of clearAllInstances(), try rebuilding IAS without clearing:

```scala
// Don't clear instances, just mark IAS dirty and rebuild with new transforms
// This would require modifying cylinder positions in-place rather than recreating
```

### 7. Investigate OptiX Cache

Check if OptiX disk cache corruption could be causing issues:

```bash
# Clear OptiX cache
rm -rf ~/.nv/ComputeCache
# Or set custom cache location
export MENGER_OPTIX_CACHE=/tmp/optix-cache-test
```

### 8. Check Stack Size

The pipeline stack size configuration might be insufficient for cylinder shaders:

```cpp
// In OptiXContext::createPipeline()
// Try increasing continuation_stack_size beyond current 8192
continuation_stack_size = std::max(continuation_stack_size, 16384u);
```

### 9. Simplify Scene Rebuild

Test if issue is specific to the full rebuild process:

```scala
// Instead of rebuilding entire scene, try:
// 1. Just clear cylinders, keep spheres/triangles
// 2. Add cylinders one at a time to identify problematic cylinder
// 3. Test with single cylinder vs many cylinders
```

## Related Code Locations

### Key Files
- `optix-jni/src/main/native/shaders/hit_cylinder.cu` - Cylinder intersection/closest-hit shaders
- `optix-jni/src/main/native/PipelineManager.cpp` - Pipeline and SBT setup
- `optix-jni/src/main/native/OptiXWrapper.cpp` - Scene management and rendering
- `menger-app/src/main/scala/menger/engines/OptiXEngine.scala` - Engine and rebuild logic
- `menger-app/src/main/scala/menger/engines/scene/TesseractEdgeSceneBuilder.scala` - Tesseract edge creation

### Key Functions
- `OptiXEngine.rebuildScene()` - Triggered on rotation (line 231)
- `OptiXWrapper::clearAllInstances()` - Clears geometry (line 974)
- `OptiXWrapper::buildPipeline()` - Rebuilds OptiX pipeline (line 490)
- `PipelineManager::buildPipeline()` - Creates modules, program groups, pipeline (line 352)
- `PipelineManager::createProgramGroups()` - Creates cylinder hitgroups (line 78)
- `OptiXWrapper::addCylinderInstance()` - Adds cylinder to scene (line 858)

## Environment

- **OS:** Linux 6.14.0-37-generic (Ubuntu)
- **CUDA Version:** (check with `nvidia-smi`)
- **OptiX Version:** (determined at runtime)
- **GPU:** (check with `nvidia-smi`)
- **Branch:** feature/sprint-8
- **Recent Commits:**
  - 0980462 - test: Add cylinder and material preset tests
  - 826ea6d - docs: Update CLI help for tesseract edge properties
  - a916a9d - refactor: Rename sphere_combined.cu to optix_shaders.cu
  - 36e7a07 - fix: Resolve cylinder module cleanup causing double-free crash
  - 153d962 - WIP: added edge-color, edge-emission and edge-radius properties

## Additional Notes

- The crash did NOT occur before cylinder edge rendering was added
- Non-cylinder geometry (spheres, triangles, tesseracts without edges) works fine with rotation
- The issue is specifically related to the cylinder cleanup and recreation cycle
- Pipeline creation itself is failing, suggesting the issue is in how modules/program groups are being set up after clearAllInstances()
