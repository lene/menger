# Troubleshooting

## OptiX JNI Build Issues

### CMake warnings "Ignoring extra path"

**Fixed** by custom `CMakeWithoutVersionBug` build tool. See `project/CMakeWithoutVersionBug.scala` - fixes sbt-jni version passing issue.

### "library not found" or "cannot find -lcuda"

- Check CUDA installed: `nvcc --version`
- Check `LD_LIBRARY_PATH`: `echo $LD_LIBRARY_PATH`
- Ubuntu/Debian: `pkexec apt-get install nvidia-cuda-toolkit`

### "OptiX headers not found"

- Set `OPTIX_ROOT` to OptiX SDK path
- CMakeLists.txt auto-detects, prefers highest version (9.0 over 8.0)
- Download: https://developer.nvidia.com/optix

### CUDA error 718 ("invalid program counter")

**Cause:** OptiX SDK/driver version mismatch

**Symptom:** `cudaDeviceSynchronize() failed: invalid program counter (718)`

**Diagnosis:**
```bash
# Check driver's OptiX version
strings /usr/lib/x86_64-linux-gnu/libnvoptix.so.* | grep "OptiX Version"

# Check SDK version used to build
grep "OptiX SDK:" optix-jni/target/native/x86_64-linux/build/CMakeCache.txt
```

**Fix:**
```bash
# Install matching OptiX SDK (9.0 for driver 580.x+)
rm -rf optix-jni/target/native
sbt "project optixJni" compile
```

**Prevention:** CMakeLists.txt auto-detects highest SDK version

**Root cause:** OptiX strict ABI compatibility - SDK must match driver runtime

### "UnsatisfiedLinkError: no optixjni in java.library.path"

- Check `optix-jni/target/native/x86_64-linux/bin/liboptixjni.so` exists
- Rebuild: `sbt "project optixJni" nativeCompile`
- For tests, `build.sbt` sets library path via `Test / javaOptions`

### CUDA Architecture Support

**Native library (`liboptixjni.so`):**
- C++ host code only (no device code)
- Calls CUDA/OptiX APIs
- Works on any system with compatible CUDA runtime

**OptiX shaders (`sphere_combined.ptx`):**
- PTX intermediate representation
- Targets compute_52 (Maxwell) minimum
- JIT-compiled at runtime to actual GPU architecture
- Single PTX works on any NVIDIA GPU 2014+ (sm_52, 75, 86, 89, etc.)
- OptiX requires one architecture target; we use virtual for compatibility

### CMake cache mismatch after Docker builds

**Cause:** CMake cache created in Docker at different path (e.g., `/builds/lilacashes/menger`)

**Automatic fix:** build.sbt detects and cleans mismatched caches

**Manual fix:**
```bash
pkexec chown -R $USER:$USER optix-jni/target/
rm -rf optix-jni/target/native
```

**Prevention:** Run Docker with your user ID (see CI_CD.md)

### "AccessDeniedException" after Docker builds

**Cause:** Docker ran as root, created root-owned files

**Fix:** `pkexec chown -R $USER:$USER optix-jni/target/`

**Prevention:** Run Docker with `--user $(id -u):$(id -g)` (see CI_CD.md)

### "Failed to open PTX file" or solid red rendering after sbt clean

**Cause:** PTX compiled to `optix-jni/target/classes/native/` but OptiX looks in `target/native/x86_64-linux/bin/`

**Symptom:** Solid red image, error `Failed to open PTX file: target/native/x86_64-linux/bin/sphere_combined.ptx (errno: 2)`

**Root cause:** After `sbt clean`, `target/` removed. Build compiles to `optix-jni/target/classes/native/` but runtime expects `target/native/x86_64-linux/bin/`

**Fix:**
```bash
mkdir -p target/native/x86_64-linux/bin
cp optix-jni/target/classes/native/x86_64-linux/sphere_combined.ptx target/native/x86_64-linux/bin/
```

### Wrong shader file being edited

**Issue:** Separate shader files (`sphere_miss.cu`, `sphere_closesthit.cu`, `sphere_raygen.cu`) NOT compiled/used

**Correct file:** Edit `sphere_combined.cu` (all shaders in one file)

**Verify:** Check `optix-jni/src/main/native/CMakeLists.txt` - specifies `shaders/sphere_combined.cu`

**Why separate files exist:** Outdated from earlier implementation

## Package Issues

### Packaged app can't find optixjni library

**Expected behavior:** Known issue - packaged app tries loading from system path not bundled location. Doesn't affect code correctness. Only impacts distribution.
