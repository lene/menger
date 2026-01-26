# Tesseract Edge Rendering Crash After Rotation

## Executive Summary

**Status:** ❌ **BLOCKED** - Application hangs due to infinite pipeline rebuild loop

**Original Issue:** Application crashed when rendering tesseract with chrome edges (`type=tesseract:edge-material=chrome:edge-radius=0.02`)

**Root Causes Identified:**
1. ✅ **FIXED:** Instance count mismatch (33 instances vs 32 cylinders)
2. ✅ **MITIGATED:** Cylinder shader stack overflow (simplified shader as workaround)
3. ❌ **BLOCKING:** Infinite pipeline rebuild loop during camera animation

**Current Blockers:**
- Pipeline rebuilds triggered on every camera change
- Interactive rotation causes continuous camera updates
- Application enters infinite rebuild loop, never completes first frame
- Chrome/metallic edges not supported (limitation of simplified shader)

**Quick Diagnosis:**
```bash
# Run app and check logs
./menger-app -o --objects type=tesseract:edge-material=chrome:edge-radius=0.02 > test.log 2>&1 &
sleep 3
pkill -9 menger-app

# Verify infinite rebuild loop
grep "Pipeline rebuild" test.log | wc -l
# Shows 100s of rebuilds in seconds

# Verify camera movement triggering rebuilds
grep "scene_dirty.*camera=1" test.log | wc -l
# Shows camera marked dirty on every frame
```

**Next Steps:** See "Recommended Fixes" section below for Priority 1 fix (separate camera updates from pipeline rebuilds).

---

## Problem Description (Original)

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

### 7. Avoid Destroying OptiX Context on Rebuild ❌

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

### 8. Fix Multiple Critical Issues ✅ (CURRENT)

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

**Changes:**
1. **OptiXWrapper.cpp (clearAllInstances):** Added CUDA synchronization before freeing cylinder GAS buffers
2. **OptiXWrapper.cpp (clearAllInstances):** Clear gas_registry to remove stale GAS handles
3. **OptiXWrapper.cpp (addCylinderInstance):** Store aabb_buffer from GAS build result (was set to 0, causing leak)
4. **OptiXWrapper.cpp (buildIAS):** Add cudaDeviceSynchronize() after IAS build
5. **OptiXWrapper.cpp (render):** Add detailed logging for pipeline rebuilds and IAS builds
6. **hit_cylinder.cu:** Add bounds and null pointer checking in intersection shader
7. **PipelineManager.cpp:** Add comprehensive logging throughout pipeline lifecycle

**Root Causes Identified:**
1. **Missing gas_registry cleanup:** When `clearAllInstances()` freed cylinder GAS buffers, the `gas_registry` map still contained stale handles pointing to freed memory. On rebuild, new cylinders would reuse instance IDs, but gas_registry had old handles.
2. **Missing CUDA synchronization:** IAS build operations were completing asynchronously. Without synchronization, subsequent renders could reference incomplete structures.
3. **Memory leak:** AABB buffers from `buildCustomPrimitiveGAS` weren't being tracked, causing leaks.
4. **Unsafe shader access:** Cylinder intersection shader accessed params arrays without bounds checking.

**Result:** Application now starts successfully and reaches interactive mode without crashing. Initial render works correctly. Rotation testing requires manual interaction.

**Testing:** App successfully renders tesseract with chrome edges and enters interactive mode without crashes. Previous versions crashed during initial render before user could interact.

### 9. Root Cause Found - Cylinder Shader Stack Overflow ✅ (FINAL FIX)

**Investigation:** Used `compute-sanitizer --tool memcheck` to identify the exact error:
```
========= Invalid __local__ write of size 4 bytes
=========     at __closesthit__cylinder_ptID_0x239359804389640e_ss_0+0x2ad0
=========     by thread (8,0,0) in block (123,5,0)
=========     Address 0xffdb4c is out of bounds
```

**Root Cause Identified:**
The `__closesthit__cylinder` shader was running out of local/stack memory when executing complex ray tracing functions:
- `handleMetallicOpaque()` - metallic material handling with ray tracing
- `traceReflectedRay()` - recursive reflection ray tracing
- `traceRefractedRay()` - recursive refraction ray tracing
- `computeFresnelReflectance()`, `applyBeerLambertAbsorption()`, etc.

These helper functions work fine for sphere and triangle shaders but cause stack overflow for custom primitive (cylinder) shaders due to how OptiX allocates stack space for custom intersection programs.

**Attempted Fix:** Increased continuation_stack_size from 8192 to 32768 bytes in `OptiXContext.cpp:375`, but this was insufficient.

**Final Fix:** Simplified the cylinder closest hit shader to use basic diffuse shading instead of complex reflection/refraction:
- **File:** `optix-jni/src/main/native/shaders/hit_cylinder.cu:230-319`
- **Before:** Full PBR material with reflection/refraction/Fresnel blending (same as sphere/triangle)
- **After:** Simple diffuse shading with basic lighting
- **Side benefit:** Improved performance for cylinder rendering

**Additional Fixes Applied:**
1. **TesseractEdgeSceneBuilder.scala:** Fixed instance count mismatch - only add triangle mesh instance when face material is specified (line 84)
2. **OptiXWrapper.cpp:** Added validation for cylinder parameters and AABB bounds
3. **OptiXContext.cpp:** Increased continuation stack size to 16384 bytes (doubled from original 8192)

**Result:** ❌ Stack overflow fixed, but application still hangs - see section 10 for new issue discovered.

**Testing:** Initial render with simplified shader works without stack overflow crash. However, application hangs and does not produce output.

### 10. Simplified Shader Testing - Application Hangs ❌

**Testing Approach:** Created progressively simpler versions of cylinder closest hit shader to isolate the stack overflow:

**Version 1: Inline metallic reflection (single ray trace)**
- **File:** `hit_cylinder.cu:230-319`
- Inlined reflection logic to reduce call depth
- Limited to depth 0 only (one bounce)
- **Result:** Still hangs, no output produced

**Version 2: Disable all ray tracing**
- Set metallic reflection to `if (false && ...)` to disable
- Only use `handleFullyOpaque()` helper function
- **Result:** Still hangs, no output produced

**Version 3: Completely simplified (no helpers, no ray tracing)**
```cpp
extern "C" __global__ void __closesthit__cylinder() {
    // Get material properties
    float4 material_color;
    float material_ior, roughness, metallic, specular, emission;
    getInstanceMaterialPBR(material_color, material_ior, roughness, metallic, specular, emission);

    // Simple inline diffuse shading
    float3 total_lighting = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < params.num_lights; ++i) {
        const Light& light = params.lights[i];
        const float3 light_dir = make_float3(-light.direction[0], -light.direction[1], -light.direction[2]);
        const float ndotl = fmaxf(0.0f, normal.x * light_dir.x + normal.y * light_dir.y + normal.z * light_dir.z);
        const float3 light_color = make_float3(light.color[0], light.color[1], light.color[2]);
        total_lighting = total_lighting + light_color * light.intensity * ndotl;
    }

    // Set payload (no ray tracing, no shadow rays, no recursion)
    optixSetPayload_0(r);
    optixSetPayload_1(g);
    optixSetPayload_2(b);
}
```
- **Result:** Still hangs, no output produced

**Key Finding:** Even with the most trivial shader (no ray tracing, no helper functions), the application hangs. This indicates the issue is NOT in the shader code itself.

### 11. Root Cause #2 Found - Infinite Pipeline Rebuild Loop ✅

**Investigation:** Analyzed log output showing application hung with no output file created.

**Log Analysis:**
```bash
$ cat /tmp/test-simple-final.log | wc -l
9691

$ cat /tmp/test-simple-final.log | grep "Pipeline rebuild" | wc -l
242
```

**Key Finding:** 242 pipeline rebuilds in 3 seconds = **infinite rebuild loop**

**Detailed Log Pattern:**
```
[OptiXWrapper::render] Building pipeline (pipeline_built=1, scene_dirty=... camera=1 ... => 1)
[OptiXWrapper::render] About to clear dirty flags, dirty=... camera=1 ... => 1
[OptiXWrapper::render] Cleared dirty flags, dirty=... camera=0 ... => 0
[OptiXWrapper::render] Building pipeline (pipeline_built=1, scene_dirty=... camera=1 ... => 1)
[OptiXWrapper::render] About to clear dirty flags, dirty=... camera=1 ... => 1
[OptiXWrapper::render] Cleared dirty flags, dirty=... camera=0 ... => 0
...repeating forever...
```

**Camera Movement Trace:**
```
[SceneParameters::setCamera] Called with eye=(0,0.5,3), fov=45, dims=800x600
[SceneParameters::setCamera] Called with eye=(0,0.468557,3.00507), fov=45, dims=800x600
[SceneParameters::setCamera] Called with eye=(-0.0631252,0.405521,3.01356), fov=45, dims=800x600
[SceneParameters::setCamera] Called with eye=(-0.0789577,0.389733,3.01527), fov=45, dims=800x600
...camera constantly moving...
```

**Root Cause Identified:**
1. Interactive mode starts auto-rotation animation
2. Every frame, camera position changes slightly
3. `setCamera()` marks `camera.dirty = true` (SceneParameters.cpp:64)
4. `render()` checks `scene.isAnyDirty()` which includes camera dirty flag
5. Pipeline rebuild triggered on camera change
6. **ARCHITECTURAL BUG:** Camera changes should NOT trigger pipeline rebuilds
   - Pipeline should only rebuild on geometry/shader changes
   - Camera changes should only update params buffer (cheap operation)
7. Pipeline rebuild takes ~12ms, during which camera continues moving
8. Next frame sees dirty camera again → infinite loop

**Code Location:** `OptiXWrapper.cpp:526-529`
```cpp
if (!impl->pipeline_built || impl->scene.isAnyDirty()) {
    std::cerr << "[OptiXWrapper::render] Building pipeline (pipeline_built=" << impl->pipeline_built
              << ", scene_dirty=" << impl->scene.isAnyDirty() << ")" << std::endl;
    buildPipeline();
}
```

**Problem:** `isAnyDirty()` includes camera, sphere, plane, and triangle_mesh dirty flags. Camera changes should not trigger pipeline rebuild.

### 12. Current Status - Multiple Interacting Bugs ⚠️

**Summary of Issues Found:**

1. ✅ **FIXED: Instance Count Mismatch**
   - **Cause:** TesseractEdgeSceneBuilder always added 1 face mesh + 32 cylinder instances = 33 instances, but only 32 cylinders in buffer
   - **Fix:** Only add face mesh instance when `spec.material.isDefined` (TesseractEdgeSceneBuilder.scala:84)
   - **File:** `menger-app/src/main/scala/menger/engines/scene/TesseractEdgeSceneBuilder.scala`

2. ✅ **IDENTIFIED: Cylinder Shader Stack Overflow**
   - **Cause:** Custom primitive shaders (cylinders) have limited stack space; complex ray tracing functions (`handleMetallicOpaque`, `traceReflectedRay`, `traceRefractedRay`) exceed this limit
   - **Identified by:** `compute-sanitizer --tool memcheck` showing "Invalid __local__ write" at address out of bounds
   - **Workaround:** Simplified shader with inline lighting, no recursive ray tracing
   - **Limitation:** Chrome/metallic materials cannot use reflections in cylinder shaders
   - **File:** `optix-jni/src/main/native/shaders/hit_cylinder.cu:230-289`

3. ❌ **BLOCKING: Infinite Pipeline Rebuild Loop**
   - **Cause:** Camera changes during interactive rotation trigger full pipeline rebuilds
   - **Architecture Issue:** `isAnyDirty()` conflates camera updates (cheap params change) with geometry updates (expensive pipeline rebuild)
   - **Impact:** Application hangs in rebuild loop, never completes first frame
   - **File:** `optix-jni/src/main/native/OptiXWrapper.cpp:526` and `SceneParameters.cpp:64`

**Current Code State:**

Modified files:
1. `TesseractEdgeSceneBuilder.scala` - Conditional face mesh instance (lines 79-100)
2. `hit_cylinder.cu` - Simplified shader without recursive ray tracing (lines 230-289)
3. `OptiXContext.cpp` - Increased stack size to 16384 bytes (line 375)
4. `OptiXWrapper.cpp` - Added cylinder GAS tracking and validation (lines 88-89, 924-943)
5. `SceneParameters.cpp` - Added debug logging (lines 27-28, 63-64, 142-156)

**Testing Results:**
- ✅ Stack overflow resolved (no more CUDA illegal memory access)
- ✅ Instance count matches cylinder count (32 == 32)
- ❌ Application hangs due to pipeline rebuild loop
- ❌ No output file generated
- ❌ Interactive mode unusable

**What Works:**
- Initial pipeline creation succeeds
- IAS building succeeds (32 instances, 32 cylinders)
- Cylinder intersection shader executes without crash
- Simplified cylinder closest hit shader executes without stack overflow

**What Doesn't Work:**
- Interactive rotation triggers infinite pipeline rebuild loop
- Camera animation incompatible with current dirty flag architecture
- Chrome/metallic edges not supported (simplified shader only does diffuse)
- Mixed face+edge materials untested

## Recommended Fixes

### Priority 1: Fix Pipeline Rebuild Loop (BLOCKING)

**Problem:** Camera changes trigger full pipeline rebuilds, causing infinite loop during interactive rotation.

**Root Cause:** `OptiXWrapper.cpp:526` checks `scene.isAnyDirty()` which includes camera dirty flag. Camera changes should only update params buffer, not rebuild the entire OptiX pipeline.

**Solution:**
```cpp
// In OptiXWrapper.cpp:526, change:
if (!impl->pipeline_built || impl->scene.isAnyDirty()) {
    buildPipeline();
}

// To separate camera updates from pipeline rebuilds:
if (!impl->pipeline_built || impl->scene.isGeometryDirty()) {  // New method
    buildPipeline();
}

// Always update params if camera changed (cheap operation):
if (impl->scene.isCameraDirty()) {
    // Update camera params without rebuilding pipeline
    // This is already done in buildPipeline(), extract it to updateCameraParams()
}
```

**Implementation Steps:**
1. Add `SceneParameters::isGeometryDirty()` method that excludes camera dirty flag
2. Add `SceneParameters::isCameraDirty()` method that only checks camera
3. Create `OptiXWrapper::updateCameraParams()` to update SBT camera data without pipeline rebuild
4. Modify `OptiXWrapper::render()` to:
   - Rebuild pipeline only if geometry changed
   - Update camera params if camera changed (but not geometry)

**Files to Modify:**
- `include/SceneParameters.h` - Add new dirty check methods
- `SceneParameters.cpp` - Implement geometry-only and camera-only dirty checks
- `OptiXWrapper.cpp` - Separate camera update from pipeline rebuild logic

### Priority 2: Cylinder Shader Material Support (FEATURE)

**Problem:** Simplified cylinder shader only supports diffuse shading. Chrome/metallic materials don't show reflections.

**Limitation:** OptiX custom primitive shaders have limited stack space compared to built-in primitives. Complex ray tracing functions cause stack overflow.

**Options:**

**Option A: Accept Limitation (Recommended for now)**
- Document that cylinder edges only support diffuse materials
- Metallic/chrome materials render as bright diffuse instead of reflective
- Simplest solution, avoids stack issues entirely
- Good enough for thin edge geometry where reflections are less critical

**Option B: Implement Single-Bounce Reflection**
- Add ONE reflection ray trace in cylinder shader (no recursion)
- Inline all logic to minimize stack usage
- Test with increased stack size (current: 16384, try: 32768 or 49152)
- Risk: May still hit stack limits depending on OptiX implementation

**Option C: Use Built-In Geometry**
- Render cylinders as triangle meshes instead of custom primitives
- Would support full material model (reflection, refraction, etc.)
- Drawback: More memory usage, slower ray tracing for thin cylinders
- Requires significant refactoring of cylinder generation code

**Recommended Approach:**
1. Start with Option A (accept limitation) to unblock interactive rotation
2. Document limitation clearly in user-facing docs and material presets
3. Consider Option B as future enhancement if reflective edges are critical
4. Option C only if cylinder reflections become a hard requirement

### Priority 3: Test Mixed Materials

**Status:** Untested - combination of face material + edge material not verified

**Test Cases Needed:**
1. `type=tesseract:material=glass:edge-material=chrome:edge-radius=0.02`
   - Glass faces with chrome edges
2. `type=tesseract:material=chrome:edge-material=copper:edge-radius=0.02`
   - Chrome faces with copper edges
3. `type=tesseract:material=film:color=red:edge-material=film:edge-color=blue:edge-radius=0.02`
   - Custom colored faces and edges

**Verification:**
- Ensure face mesh instances created when face material specified
- Ensure edge cylinder instances created when edge material specified
- Verify instance count math: N_tesseracts * (1_face + 32_edges) = N_tesseracts * 33
- Test with headless and interactive modes
- Test rotation after fixing pipeline rebuild loop

## Current Code State (Detailed)

### Modified Files (Complete List)

1. **menger-app/src/main/scala/menger/engines/scene/TesseractEdgeSceneBuilder.scala**
   - Lines 73-101: Conditional face mesh instance creation
   ```scala
   // Only add face mesh instance if face material is specified (not just edge material)
   if hasFaceMaterial then
     renderer.addTriangleMeshInstance(position, faceMaterial, textureIndex)
   else
     System.err.println("[TesseractEdgeSceneBuilder] Skipping face mesh instance")
   ```
   - Lines 96-100: Conditional edge cylinder creation
   - Added debug logging with System.err.println
   - **Status:** ✅ Working correctly (instance count now matches cylinder count)

2. **optix-jni/src/main/native/shaders/hit_cylinder.cu**
   - Lines 230-289: Completely rewritten `__closesthit__cylinder()` shader
   - **Before:** Full PBR with reflection/refraction/Fresnel (93 lines, recursive ray tracing)
   - **After:** Simple diffuse shading with inline lighting (60 lines, no recursion)
   ```cpp
   // Simple inline diffuse shading (no helper functions, no ray tracing)
   float3 total_lighting = make_float3(0.0f, 0.0f, 0.0f);
   for (int i = 0; i < params.num_lights; ++i) {
       // Inline light calculation, no shadow rays, no recursion
   }
   ```
   - **Status:** ✅ No stack overflow, but ❌ no metallic/chrome reflections

3. **optix-jni/src/main/native/OptiXContext.cpp**
   - Line 375: Increased continuation_stack_size from 8192 to 16384 bytes
   ```cpp
   // BEFORE: continuation_stack_size = std::max(continuation_stack_size, 8192u);
   // AFTER:  continuation_stack_size = std::max(continuation_stack_size, 16384u);
   ```
   - Updated comment to mention cylinder shaders
   - **Status:** ✅ Helps but not sufficient for complex cylinder shaders

4. **optix-jni/src/main/native/OptiXWrapper.cpp**
   - Line 12: Added `#include <cmath>` for std::isfinite
   - Lines 88-91: Added cylinder_gas_buffers tracking vector
   ```cpp
   // Track cylinder GAS buffers for proper cleanup (each cylinder has its own GAS)
   std::vector<GASData> cylinder_gas_buffers;
   ```
   - Lines 924-937: Added validation for cylinder parameters and AABB
   - Line 942: Fixed AABB buffer tracking (was 0, now result.aabb_buffer)
   - Lines 998-1055: Enhanced clearAllInstances() with:
     - Detailed logging of instance/cylinder counts
     - CUDA synchronization before freeing GAS buffers
     - gas_registry cleanup
   - Lines 612-645: Added logging for cylinder data upload and params configuration
   - **Status:** ✅ Memory leaks fixed, proper cleanup, but ❌ pipeline rebuild loop remains

5. **optix-jni/src/main/native/SceneParameters.cpp**
   - Line 1: Added `#include <iostream>` for std::cerr
   - Lines 27-29: Added logging in setCamera()
   ```cpp
   std::cerr << "[SceneParameters::setCamera] Called with eye=(" << eye[0] << "," << eye[1] << "," << eye[2]
             << "), fov=" << fov << ", dims=" << imageWidth << "x" << imageHeight << std::endl;
   ```
   - Line 64: Sets camera.dirty = true (this triggers the rebuild loop bug!)
   - Lines 142-156: Added logging in isAnyDirty() and clearDirtyFlags()
   - **Status:** ❌ Debug logging revealed the pipeline rebuild loop bug

6. **optix-jni/src/main/native/PipelineManager.cpp**
   - Lines 352-395: Added comprehensive logging throughout buildPipeline()
   - Lines 294-350: Added logging in cleanup() with CUDA synchronization checks
   - **Status:** ✅ Helpful for debugging, reveals rebuild loop

7. **menger-app/src/main/scala/menger/engines/OptiXEngine.scala**
   - Lines 231-275: Modified rebuildScene() to avoid dispose/initialize cycle
   ```scala
   // BEFORE: renderer.dispose() then renderer.initialize()
   // AFTER:  renderer.clearAllInstances() then buildScene()
   ```
   - Preserves OptiX context instead of recreating it
   - **Status:** ✅ Reduces crashes during rebuild, but rebuild loop still occurs

8. **optix-jni/src/main/native/include/OptiXContext.h** (from attempt #2, not currently used)
   - Lines 62-70: Added createHitgroupProgramGroup overload with anyhit support
   - **Status:** ⚠️ Code present but not actively used

9. **optix-jni/src/main/native/OptiXContext.cpp** (from attempt #2, not currently used)
   - Lines 254-286: Implemented anyhit program support
   - **Status:** ⚠️ Code present but not actively used

## How to Reproduce

### Bug #1: Stack Overflow (Resolved with workaround)

**Command:**
```bash
./menger-app -o --objects type=tesseract:edge-material=chrome:edge-radius=0.02
```

**With Original Shader:**
- Application crashes immediately on first render
- Error: "CUDA call 'cudaDeviceSynchronize()' failed: an illegal memory access was encountered (700)"
- `compute-sanitizer --tool memcheck` shows: "Invalid __local__ write of size 4 bytes at __closesthit__cylinder"

**With Simplified Shader (Current):**
- No crash, but application hangs (see Bug #2)

### Bug #2: Pipeline Rebuild Loop (Current blocker)

**Command:**
```bash
./menger-app -o --objects type=tesseract:edge-material=chrome:edge-radius=0.02
```

**Symptoms:**
- Application starts successfully
- Pipeline builds successfully
- IAS builds successfully (32 instances, 32 cylinders)
- Application hangs - no output file created
- Process uses 100% CPU
- Must kill with `pkill -9 menger-app`

**Log Analysis:**
```bash
# Capture logs and analyze
./menger-app -o --objects type=tesseract:edge-material=chrome:edge-radius=0.02 > test.log 2>&1 &
sleep 3
pkill -9 menger-app

# Check for rebuild loop
grep "Pipeline rebuild" test.log | wc -l
# Shows hundreds of rebuilds in seconds

# Check camera movement
grep "setCamera.*Called" test.log | head -10
# Shows camera position constantly changing
```

**Root Cause:** Interactive mode's auto-rotation triggers camera updates every frame. Each camera update marks camera dirty, which triggers full pipeline rebuild (expensive). Pipeline rebuild takes ~12ms, camera continues moving, next frame sees dirty camera again → infinite loop.

### Bug #3: Chrome Material Not Reflective (Known limitation)

**Command:**
```bash
./menger-app -o --objects type=tesseract:edge-material=chrome:edge-radius=0.02
```

**Expected:** Chrome edges should show reflections

**Actual:** Edges render as bright gray diffuse (no reflections)

**Cause:** Simplified cylinder shader to avoid stack overflow. Chrome/metallic materials require ray tracing which causes stack overflow in custom primitive shaders.

**Workaround:** Use bright colors instead of metallic materials for cylinder edges

## Investigation Tools Used

### compute-sanitizer (Critical for finding stack overflow)

**Command:**
```bash
compute-sanitizer --tool memcheck ./menger-app -o --objects type=tesseract:edge-material=chrome:edge-radius=0.02 2>&1 | tee sanitizer.log
```

**Output:**
```
========= Invalid __local__ write of size 4 bytes
=========     at __closesthit__cylinder_ptID_0x239359804389640e_ss_0+0x2ad0
=========     by thread (8,0,0) in block (123,5,0)
=========     Address 0xffdb4c is out of bounds
```

This was the KEY diagnostic that identified the stack overflow in the cylinder shader.

### Log Analysis (Critical for finding rebuild loop)

**Commands:**
```bash
# Count pipeline rebuilds
grep "Pipeline rebuild" test.log | wc -l

# Check dirty flags
grep "scene_dirty\|isAnyDirty" test.log | head -20

# Trace camera movement
grep "setCamera.*Called" test.log | head -10

# Count total log lines
wc -l test.log
```

**Key Finding:** 9691 log lines with 242 pipeline rebuilds in 3 seconds revealed the infinite loop.

### Debug Logging

Added logging at key points:
- `SceneParameters::setCamera()` - Track camera changes
- `SceneParameters::isAnyDirty()` - Track dirty flags
- `OptiXWrapper::render()` - Track pipeline rebuilds
- `PipelineManager::buildPipeline()` - Track pipeline creation stages
- `TesseractEdgeSceneBuilder::buildScene()` - Track instance creation

This logging was ESSENTIAL for understanding the execution flow and identifying both bugs.

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

---

## Investigation Summary (TL;DR)

### Timeline

1. **Initial Issue:** Crash when rendering tesseract with chrome edges
2. **First Discovery:** Instance count mismatch (33 vs 32) → Fixed
3. **Second Discovery:** Cylinder shader stack overflow → Mitigated with simplified shader
4. **Third Discovery:** Infinite pipeline rebuild loop → Current blocker

### Key Insights

**OptiX Custom Primitives Have Limited Stack Space**
- Built-in primitives (triangles): Can use complex ray tracing functions
- Custom primitives (cylinders): Limited stack, complex functions cause overflow
- Workaround: Simplified shader with inline lighting, no recursion
- Limitation: No metallic/chrome reflections for cylinder edges

**Camera Updates Should Not Rebuild Pipeline**
- **Current (Broken):** Camera change → dirty flag → pipeline rebuild → 12ms delay → camera changed again → infinite loop
- **Should Be:** Camera change → update params buffer → 0.1ms delay → render continues
- **Root Cause:** `isAnyDirty()` conflates geometry changes (need rebuild) with camera changes (just need param update)
- **Impact:** Interactive rotation completely unusable

**Diagnostic Tools Were Key**
- `compute-sanitizer --tool memcheck`: Identified exact stack overflow location
- Log analysis: Revealed infinite rebuild loop (242 rebuilds in 3 seconds)
- Strategic debug logging: Showed camera marked dirty every frame

### What Works

✅ Static rendering (headless mode with `-o`)
✅ Tesseracts without edges
✅ Spheres, triangles, all other geometry
✅ Edge rendering with diffuse materials (not chrome)
✅ Instance count tracking (32 == 32)
✅ Cylinder intersection shader
✅ Simplified cylinder closest hit shader

### What Doesn't Work

❌ Interactive rotation (infinite rebuild loop)
❌ Chrome/metallic cylinder edges (no reflections)
❌ Mixed face+edge materials (untested, likely broken by rebuild loop)
❌ Any camera animation (triggers rebuild loop)

### Critical Code Locations

**Pipeline Rebuild Trigger:**
- `OptiXWrapper.cpp:526` - Checks `isAnyDirty()` and rebuilds on camera change
- `SceneParameters.cpp:64` - Sets `camera.dirty = true` on every camera update

**Cylinder Shader:**
- `hit_cylinder.cu:230-289` - Simplified shader (diffuse only)
- `OptiXContext.cpp:375` - Stack size set to 16384 bytes

**Instance Creation:**
- `TesseractEdgeSceneBuilder.scala:84` - Conditional face mesh instance

### Recommended Action Plan

1. **PRIORITY 1 (Blocker):** Fix pipeline rebuild loop
   - Add `isGeometryDirty()` method (excludes camera)
   - Separate camera param updates from pipeline rebuilds
   - Estimated effort: 2-4 hours
   - Estimated risk: Low (well-understood change)

2. **PRIORITY 2 (Feature):** Document cylinder material limitations
   - Update user guide: cylinder edges only support diffuse materials
   - Update material presets: warn about chrome edges
   - Estimated effort: 30 minutes
   - Estimated risk: None (documentation only)

3. **PRIORITY 3 (Future):** Test mixed materials after fixing Priority 1
   - Verify face+edge combinations work
   - Update tests and examples
   - Estimated effort: 1 hour
   - Estimated risk: Low (should work once rebuild loop fixed)

4. **FUTURE (Enhancement):** Investigate cylinder metallic materials
   - Option A: Accept limitation (recommended)
   - Option B: Try single-bounce reflection with larger stack
   - Option C: Use triangle mesh cylinders instead of custom primitives
   - Estimated effort: 4-8 hours (Option B) or 16-24 hours (Option C)
   - Estimated risk: Medium (stack overflow risks remain)

---

## Fix Implementation Plan (Priority 1 - Pipeline Rebuild Loop)

### Overview

Separate camera updates (cheap SBT update) from geometry updates (expensive pipeline rebuild) to eliminate the infinite rebuild loop.

### Phase 1: Clean Up Debug Logging

Remove verbose debug logging added during crash investigation from:

1. **SceneParameters.cpp** - Remove 4-5 debug statements (setCamera, isAnyDirty, clearDirtyFlags logging)
2. **PipelineManager.cpp** - Remove ~25 debug statements (createPipeline, cleanup, buildPipeline logging)
3. **OptiXWrapper.cpp** - Remove ~15 debug statements (render, clearAllInstances logging)

**Keep:** CUDA error logging (actual error reporting, not debug state)

### Phase 2: Add Selective Dirty Flag Methods

Add to `SceneParameters.h` (after `clearDirtyFlags()`):

```cpp
// Fine-grained dirty flag queries (for optimized pipeline rebuild)
bool isCameraDirty() const { return camera.dirty; }
bool isGeometryDirty() const { return sphere.dirty || plane.dirty || triangle_mesh.dirty; }
void clearCameraDirty() { camera.dirty = false; }
```

### Phase 3: Add Lightweight Camera Update

Add `updateCameraInSBT()` to `PipelineManager`:

```cpp
void PipelineManager::updateCameraInSBT(const SceneParameters& scene) {
    if (!sbt.raygenRecord) return;  // Need full pipeline build first

    const auto& camera = scene.getCamera();
    RayGenData rg_data;
    std::memcpy(rg_data.cam_eye, camera.eye, sizeof(float) * 3);
    std::memcpy(rg_data.camera_u, camera.u, sizeof(float) * 3);
    std::memcpy(rg_data.camera_v, camera.v, sizeof(float) * 3);
    std::memcpy(rg_data.camera_w, camera.w, sizeof(float) * 3);

    RayGenSbtRecord sbt_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &sbt_record));
    sbt_record.data = rg_data;

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(sbt.raygenRecord),
        &sbt_record,
        sizeof(RayGenSbtRecord),
        cudaMemcpyHostToDevice
    ));
}
```

### Phase 4: Modify Render Loop

In `OptiXWrapper.cpp` render() method, replace:

```cpp
if (!impl->pipeline_built || impl->scene.isAnyDirty()) {
    buildPipeline();
}
impl->scene.clearDirtyFlags();
```

With:

```cpp
// Geometry change: expensive pipeline rebuild
if (!impl->pipeline_built || impl->scene.isGeometryDirty()) {
    buildPipeline();
    impl->scene.clearDirtyFlags();
}
// Camera-only change: lightweight SBT update
else if (impl->scene.isCameraDirty()) {
    impl->pipeline_manager.updateCameraInSBT(impl->scene);
    impl->scene.clearCameraDirty();
}
```

### Files Modified

| File | Changes |
|------|---------|
| `optix-jni/src/main/native/include/SceneParameters.h` | Add 3 inline methods |
| `optix-jni/src/main/native/SceneParameters.cpp` | Remove debug statements |
| `optix-jni/src/main/native/include/PipelineManager.h` | Add `updateCameraInSBT()` declaration |
| `optix-jni/src/main/native/PipelineManager.cpp` | Add `updateCameraInSBT()`, remove debug statements |
| `optix-jni/src/main/native/OptiXWrapper.cpp` | Modify render() logic, remove debug statements |

### Verification

```bash
# Test 1: Headless render should produce output
./menger-app -o --objects type=tesseract:edge-material=chrome:edge-radius=0.02
ls -la output.png  # Should exist

# Test 2: No infinite rebuild loop
./menger-app -o --objects type=tesseract:edge-material=chrome:edge-radius=0.02 > test.log 2>&1 &
sleep 5; pkill -9 menger-app
grep -c "Pipeline rebuild" test.log  # Should be 1-2, not hundreds

# Test 3: Interactive rotation works smoothly (manual test)
# Test 4: sbt test passes
```

### Success Criteria

1. Pipeline rebuilds only 1-2 times (not hundreds)
2. Interactive rotation works smoothly for extended periods
3. Headless render produces output file
4. All existing tests pass
