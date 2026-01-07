# Issue: Stale PTX Files Loaded After Shader Recompilation

**Priority**: High
**Type**: Bug
**Component**: Build System / OptiX JNI

## Problem Description

The OptiX renderer loads PTX (compiled CUDA shader) files from multiple locations with a search path priority. After recompiling shaders, **stale/old PTX files are frequently loaded** instead of the newly compiled version, causing:

1. Shaders to execute old code despite source changes
2. Confusing debugging sessions where fixes don't take effect
3. Need for manual PTX file copying between directories
4. Wasted development time tracking down "phantom bugs"

## Root Cause

OptiXWrapper.cpp searches for PTX files in this priority order:

```cpp
std::vector<std::string> ptx_search_paths = {
    "target/native/x86_64-linux/bin/sphere_combined.ptx",          // #1: Extracted from JAR
    "optix-jni/target/native/x86_64-linux/bin/sphere_combined.ptx", // #2: sbt build output
    "optix-jni/target/classes/native/x86_64-linux/sphere_combined.ptx"  // #3: sbt-jni managed
};
```

**Problem**: Location #1 (`target/native/x86_64-linux/bin/`) often contains stale PTX files that don't get automatically updated when shaders are recompiled via `sbt nativeCompile`.

## Symptoms

- Running `sbt nativeCompile` compiles shaders successfully
- New PTX generated at `optix-jni/target/native/x86_64-linux/bin/sphere_combined.ptx`
- Tests/demos still use old shader behavior
- Manual `cp optix-jni/target/.../sphere_combined.ptx target/.../` required
- No error messages - just silently wrong behavior

## Reproduction

1. Modify shader source: `optix-jni/src/main/native/shaders/sphere_combined.cu`
2. Run: `sbt "project optixJni" nativeCompile`
3. Run tests: `sbt test`
4. Observe: Tests use old shader code
5. Run: `cp optix-jni/target/native/x86_64-linux/bin/sphere_combined.ptx target/native/x86_64-linux/bin/`
6. Run tests again: Tests now use new shader code

## Impact

**During this session**:
- Multi-light implementation appeared broken for 1+ hour
- Root cause was stale PTX file (missing multi-light loop code)
- Required multiple debug iterations and manual PTX copying
- Similar issue occurred during previous shadow implementation work

## Proposed Solutions

### Option 1: Single Canonical PTX Location (Recommended)
- Store PTX in ONE location only: `optix-jni/target/native/x86_64-linux/bin/`
- Update OptiXWrapper.cpp search paths to prioritize build output
- Remove/ignore `target/native/x86_64-linux/bin/` directory entirely
- Add to .gitignore if not already present

**Pros**: Simple, matches build tool output location, no copying
**Cons**: May require changes to packaging for JAR distribution

### Option 2: Automatic PTX Synchronization
- Add sbt task that automatically copies PTX after native compilation
- Hook into `nativeCompile` task to sync to all search locations
- Ensure `sbt compile` triggers full PTX sync

**Pros**: Maintains existing search path logic
**Cons**: More complex, still involves file copying

### Option 3: PTX Timestamp Validation
- Modify OptiXWrapper.cpp to check file timestamps
- Load most recent PTX file instead of first found
- Add warning if multiple PTX versions detected

**Pros**: Detects stale files, provides debugging info
**Cons**: More complex C++ code, doesn't prevent confusion

### Option 4: Fail-Fast on Stale PTX
- Calculate checksum/hash of PTX at build time
- Embed expected hash in code
- Fail loudly if loaded PTX doesn't match expected version

**Pros**: Catches problem immediately
**Cons**: Requires build system integration, may be overkill

## Recommended Approach

**Combination of Options 1 + 3**:

1. **Primary**: Change search path priority to prefer build output:
   ```cpp
   std::vector<std::string> ptx_search_paths = {
       "optix-jni/target/native/x86_64-linux/bin/sphere_combined.ptx",  // Build output first
       "optix-jni/target/classes/native/x86_64-linux/sphere_combined.ptx",
       "target/native/x86_64-linux/bin/sphere_combined.ptx"  // JAR extraction last
   };
   ```

2. **Safety**: Add timestamp warning:
   ```cpp
   if (multiple PTX files found && timestamps differ > 60 seconds) {
       std::cerr << "WARNING: Multiple PTX files found with different ages:\n";
       std::cerr << "  " << path1 << " (" << age1 << " seconds old)\n";
       std::cerr << "  " << path2 << " (" << age2 << " seconds old)\n";
       std::cerr << "Using: " << selected_path << "\n";
   }
   ```

3. **Documentation**: Update docs/TROUBLESHOOTING.md with PTX loading behavior

## Additional Notes

- This is a **recurring problem** that has caused issues multiple times
- Affects development velocity significantly
- Silent failures are dangerous - easy to ship broken code
- Related to OptiX-specific build complexity (PTX intermediate format)

## References

- OptiXWrapper.cpp:233-242 (loadPTXModules function)
- Previous instance: During shadow ray implementation
- Current instance: During multi-light implementation (2025-11-18)

## Success Criteria

After fix is implemented:
- [ ] Modifying shader source and running `sbt nativeCompile` always uses new shader
- [ ] No manual PTX file copying required
- [ ] Clear warning if stale PTX detected
- [ ] Documented in TROUBLESHOOTING.md
- [ ] Added to test suite (verify PTX freshness)
