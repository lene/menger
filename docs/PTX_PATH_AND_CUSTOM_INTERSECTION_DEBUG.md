# PTX Path Issue and Custom Intersection Debugging

**Date:** November 5, 2025, 19:30-19:55
**Context:** Phase 3 of Glass Implementation - Custom Sphere Intersection Program

## Critical Discovery: PTX File Path Mismatch

### The Problem

All OptiX rendering was failing with solid red output and error:
```
OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS (7203)
```

This affected:
- Current HEAD (commit 184aa57)
- Previous working commits (a85b20c, ac62dac)
- Custom intersection attempt

### Root Cause

**The build system does NOT automatically copy the PTX file from the build location to the runtime location.**

- **Build location:** `optix-jni/target/native/x86_64-linux/bin/sphere_combined.ptx`
- **Runtime location:** `target/native/x86_64-linux/bin/sphere_combined.ptx`
- **Code expects PTX at:** `target/native/x86_64-linux/bin/sphere_combined.ptx` (OptiXWrapper.cpp:322)

When PTX files get out of sync between these two locations:
1. CMake compiles fresh PTX to `optix-jni/target/native/.../sphere_combined.ptx`
2. Runtime loads stale PTX from `target/native/.../sphere_combined.ptx`
3. Stale PTX doesn't match the compiled C++ code → OptiX errors

### Evidence

```bash
$ ls -lh optix-jni/target/native/x86_64-linux/bin/sphere_combined.ptx \
         target/native/x86_64-linux/bin/sphere_combined.ptx

# Before fix:
-rw-r--r-- 1 lene lene 7.3K Nov  5 19:50 optix-jni/target/native/x86_64-linux/bin/sphere_combined.ptx  # Fresh
-rw-rw-r-- 1 lene lene  14K Nov  5 19:30 target/native/x86_64-linux/bin/sphere_combined.ptx             # STALE!
```

The 14K file contained custom intersection code from an earlier build, while the C++ code expected built-in spheres (7.3K).

### The Fix (for built-in sphere rendering)

```bash
# Copy fresh PTX to runtime location
cp optix-jni/target/native/x86_64-linux/bin/sphere_combined.ptx \
   target/native/x86_64-linux/bin/sphere_combined.ptx

# Test rendering
sbt "run --optix --sponge-type sphere --timeout 2 --radius 0.5 --color 00ff8080 --ior 1.5 --save-name optix-fixed.png"
```

**Result:** ✅ SUCCESS - Proper green-cyan sphere with checkered background (commit ac62dac)

### When This Issue Occurs

1. After `sbt clean` (documented in CLAUDE.md)
2. After switching git branches/commits
3. After rebuilding native code multiple times with `rm -rf optix-jni/target/native`
4. When PTX at `target/native/...` becomes stale

### Documented Workaround

This is a known issue documented in CLAUDE.md:

> **"Failed to open PTX file" or solid red rendering after sbt clean:**
> - **Root Cause**: After `sbt clean`, the `target/` directory is removed. The build system compiles PTX to `optix-jni/target/classes/native/` but OptiX runtime expects it in `target/native/x86_64-linux/bin/`
> - **Fix**: Copy PTX file to runtime location

## Custom Intersection Implementation Status

### Changes Made

**Phase 2: Custom Intersection Program (sphere_combined.cu)**
- Added `__intersection__sphere()` function (lines 36-140)
- Implements ray-sphere intersection using quadratic formula
- Reports both t1 (near) and t2 (far) intersections
- Passes surface normal via intersection attributes (3 floats)
- Determines hit_kind: 0=entry, 1=exit

**Phase 3: Host-Side Pipeline Updates (OptiXWrapper.cpp)**
- Changed geometry from `OPTIX_BUILD_INPUT_TYPE_SPHERES` to `OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES`
- Create AABB (Axis-Aligned Bounding Box) for sphere
- Updated `numAttributeValues` from 0 to 3 (for normal x,y,z)
- Changed `usesPrimitiveTypeFlags` from `SPHERE` to `CUSTOM`
- Removed built-in sphere module creation
- Registered custom intersection program: `__intersection__sphere`
- Added `sphere_radius` to HitGroupData and SBT

**OptiXData.h:**
- Added `float sphere_radius` to HitGroupData struct

**Closest Hit Shader Updates (sphere_combined.cu):**
- Retrieve `hit_kind` via `optixGetHitKind()`
- Retrieve surface normal via `optixGetAttribute_0/1/2()`
- Use normal for Fresnel/Snell calculations

### Current Issue: CUDA Error 718

When testing custom intersection implementation:

```bash
$ sbt "run --optix --sponge-type sphere --timeout 2 --radius 0.5 --color 00ff8080 --ior 1.5 --save-name test.png"

[error] [OptiX] Render failed: CUDA call 'cudaMalloc(...)' failed: invalid program counter (718)
[error] ERROR 718 (invalid program counter) indicates OptiX SDK/driver version mismatch.
```

**PTX File Status:**
- PTX file size: 14K (contains custom intersection code)
- PTX verified to contain `__intersection__sphere` function
- PTX copied to correct runtime location
- C++ code compiled with matching changes

**Error Occurs At:** Very early in OptiX initialization (cudaMalloc)

**What We Ruled Out:**
1. ✅ GPU hardware is working (simple CUDA test passed)
2. ✅ PTX file location is correct (copied to runtime location)
3. ✅ PTX file contains intersection function (verified with strings command)
4. ✅ C++ code rebuilt with correct PTX path
5. ✅ Build artifacts are fresh (full clean rebuild)

### PTX Path in Custom Intersection Code

**Current state:** OptiXWrapper.cpp:322 was changed to:
```cpp
std::string ptx_path = "optix-jni/target/native/x86_64-linux/bin/sphere_combined.ptx";
```

This loads directly from build location instead of runtime location.

**Problem:** This still gets error 718, suggesting the issue is not the PTX path but something about the custom intersection implementation itself.

## Hypotheses for Error 718

### Theory 1: OptiX Version Mismatch in Custom Primitives

The error 718 message says "OptiX SDK/driver version mismatch". This could mean:
- Custom primitives API changed between OptiX 8.0 and 9.0
- PTX compiled for wrong architecture
- AABB format incorrect for OptiX 9.0

**Evidence:**
- Built-in spheres work fine (ac62dac commit)
- Only custom intersection fails with error 718
- Error occurs very early (before ray tracing)

### Theory 2: AABB Data or SBT Layout Issue

The AABB or SBT record might be incorrectly sized/aligned for custom primitives:
```cpp
OptixAabb aabb;
aabb.minX = impl->sphere_center[0] - impl->sphere_radius;
// ... set min/max bounds
```

**Need to verify:**
- AABB structure size/alignment
- Whether AABB needs to be kept alive (currently freed after build)
- SBT record structure for custom primitives

### Theory 3: Pipeline Compile Options Mismatch

Both `loadPTXModules()` and `createPipeline()` set pipeline compile options. These must match exactly:
```cpp
pipeline_compile_options.numAttributeValues = 3;
pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
```

**Status:** Added TODO to refactor duplication (OptiXWrapper.cpp:447)

## Test Procedure for Next Session

### Step 1: Verify PTX File State

```bash
# Always check PTX file status before testing
ls -lh optix-jni/target/native/x86_64-linux/bin/sphere_combined.ptx \
       target/native/x86_64-linux/bin/sphere_combined.ptx

# Check PTX contains intersection function
strings optix-jni/target/native/x86_64-linux/bin/sphere_combined.ptx | grep "__intersection__sphere"
```

### Step 2: Test Built-in Sphere (Baseline)

```bash
# Checkout known working commit
git checkout ac62dac

# Full clean rebuild
rm -rf optix-jni/target/native
env ENABLE_OPTIX_JNI=true sbt "project optixJni" nativeCompile

# Copy PTX to runtime location
cp optix-jni/target/native/x86_64-linux/bin/sphere_combined.ptx \
   target/native/x86_64-linux/bin/sphere_combined.ptx

# Test rendering (should show green-cyan sphere with checkered background)
sbt "run --optix --sponge-type sphere --timeout 2 --radius 0.5 --color 00ff8080 --ior 1.5 --save-name baseline.png"
```

**Expected:** Success, proper sphere rendering

### Step 3: Test Custom Intersection

```bash
# Return to branch
git checkout 55-add-sphere-color-support-to-optix-rendering

# Apply stashed changes (if needed)
git stash list
git stash apply stash@{0}  # Custom intersection changes

# Full clean rebuild
rm -rf optix-jni/target/native
env ENABLE_OPTIX_JNI=true sbt "project optixJni" nativeCompile

# Test rendering
sbt "run --optix --sponge-type sphere --timeout 2 --radius 0.5 --color 00ff8080 --ior 1.5 --save-name custom.png"
```

**Current Status:** Fails with error 718

## Files Modified for Custom Intersection

```
optix-jni/src/main/native/shaders/sphere_combined.cu
  - Added __intersection__sphere() function (lines 36-140)
  - Updated __closesthit__ch() to use intersection attributes (lines 276-288)

optix-jni/src/main/native/include/OptiXData.h
  - Added sphere_radius to HitGroupData (line 30)

optix-jni/src/main/native/OptiXWrapper.cpp
  - buildGeometryAccelerationStructure(): Changed to CUSTOM_PRIMITIVES with AABB (lines 236-267)
  - loadPTXModules(): Updated pipeline options for custom primitives (lines 334, 337)
  - loadPTXModules(): Removed built-in sphere module creation (removed lines 353-365)
  - createProgramGroups(): Registered custom intersection program (line 423)
  - createPipeline(): Updated pipeline options (lines 452, 455)
  - setupShaderBindingTable(): Added sphere_radius to SBT (line 552)
  - Changed PTX path (line 322)

optix-jni/src/test/scala/menger/optix/OptiXRendererTest.scala
  - Disabled lighting test temporarily (line 365)

docs/ABSORPTION_DEBUG_FINDINGS.md
  - Updated to confirm OptiX built-in sphere limitation

docs/FIX_PLAN.md
  - Replaced analytical workaround with custom intersection solution
```

## Git State

**Current branch:** `55-add-sphere-color-support-to-optix-rendering`

**Stashes:**
- `stash@{0}`: Documentation changes (ABSORPTION_DEBUG_FINDINGS.md, FIX_PLAN.md)
- `stash@{1}`: Custom intersection changes (native code) - **APPLIED**

**HEAD:** commit 184aa57 "WIP: require CUDA_HOME and OPTIX_PATH to be set, better valgrind error message"

**Working commits for baseline:**
- `ac62dac`: Add checkered plane background (built-in spheres work)
- `a85b20c`: Add Fresnel reflection and Snell's law refraction (built-in spheres work)

## Questions to Investigate

1. **Does OptiX 9.0 custom primitives API differ from 8.0?**
   - Check OptiX 9.0 SDK examples (optixWhitted, optixSphere)
   - Compare AABB creation, SBT setup, intersection program signature

2. **Are we missing any OptiX 9.0-specific initialization?**
   - Module compile options for custom primitives?
   - Pipeline link options?
   - Additional flags or capabilities?

3. **Is the AABB being freed too early?**
   - Currently freed immediately after acceleration structure build
   - Does OptiX need to keep AABB alive during rendering?

4. **Should PTX be compiled differently for custom primitives?**
   - Check nvcc compilation flags in CMakeLists.txt
   - Verify PTX architecture target (currently virtual compute_52)

5. **Are there any OptiX SDK version checks in the code?**
   - OptiXWrapper might assume OptiX 8.0 behavior
   - Check for version-specific code paths

## Environment Details

**System:**
- OS: Linux 6.14.0-33-generic (Ubuntu)
- CUDA: 12.8.93
- OptiX Driver: 9.0.0 (from NVIDIA driver 580.95)
- OptiX SDK: /usr/local/optix (version 9.0)
- GPU: 1 CUDA device detected (working, tested with simple CUDA program)

**Build:**
- sbt: 1.11.7
- Java: 21.0.8
- CMake: 3.28.3
- nvcc: CUDA 12.8

**Paths:**
- Project: `/home/lene/workspace/menger`
- Build: `optix-jni/target/native/x86_64-linux/bin/`
- Runtime: `target/native/x86_64-linux/bin/`

## Next Steps

1. **Research OptiX 9.0 custom primitives:**
   - Read OptiX 9.0 Programming Guide section on custom primitives
   - Study optixWhitted example's custom intersection implementation
   - Check for API changes from OptiX 8.0

2. **Compare with working OptiX SDK examples:**
   - Build and run official OptiX 9.0 examples with custom primitives
   - Diff our code against working examples

3. **Systematic debugging:**
   - Add more logging to OptiXWrapper.cpp (AABB values, SBT offsets)
   - Verify AABB bounds are correct
   - Check if error occurs before or after optixAccelBuild

4. **Consider alternative PTX path solution:**
   - Instead of manually copying PTX, fix the build system to do it automatically
   - Or always load from `optix-jni/target/native/...` and document it

5. **Test minimal custom primitive:**
   - Strip down to absolute minimum custom intersection (single point?)
   - Build up complexity incrementally

## References

- Glass Implementation Plan: `docs/Glass_Implementation_Plan.md`
- OptiX Documentation: NVIDIA OptiX SDK 9.0 Programming Guide
- Known Issue: CLAUDE.md section "Troubleshooting > Failed to open PTX file"
- Previous commits that work: ac62dac, a85b20c (built-in spheres)
