← [User Guide Index](USER_GUIDE.md)

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

### Stale PTX files loaded after shader recompilation

**Symptom:** Shader changes don't take effect even after `sbt nativeCompile`

**Cause:** OptiXWrapper.cpp searches multiple PTX locations in priority order:
1. `target/native/x86_64-linux/bin/sphere_combined.ptx` (extracted from JAR, often stale)
2. `optix-jni/target/native/x86_64-linux/bin/sphere_combined.ptx` (fresh build output)
3. `optix-jni/target/classes/native/x86_64-linux/sphere_combined.ptx` (sbt-jni managed)

If location #1 contains a stale PTX file, it gets loaded instead of your fresh compilation.

**Fix (immediate):**
```bash
# Force-sync fresh PTX to priority location
cp optix-jni/target/native/x86_64-linux/bin/sphere_combined.ptx target/native/x86_64-linux/bin/
```

**Fix (cleanup):**
```bash
# Remove stale PTX files
rm -f target/native/x86_64-linux/bin/sphere_combined.ptx
# Rebuild will now use fresh PTX from optix-jni/target/
```

**Prevention:** After modifying shaders, always check timestamps:
```bash
ls -lah */target/native/x86_64-linux/bin/sphere_combined.ptx target/native/x86_64-linux/bin/sphere_combined.ptx
```

**Note:** See `docs/archive/PTX_LOADING_ISSUE.md` for detailed analysis and proposed long-term solutions.

### Wrong shader file being edited

**Issue:** Separate shader files (`sphere_miss.cu`, `sphere_closesthit.cu`, `sphere_raygen.cu`) NOT compiled/used

**Correct file:** Edit `sphere_combined.cu` (all shaders in one file)

**Verify:** Check `optix-jni/src/main/native/CMakeLists.txt` - specifies `shaders/sphere_combined.cu`

**Why separate files exist:** Outdated from earlier implementation

### SIGBUS crash in libnvidia-glcore.so during window cleanup

**Symptom:** JVM crashes with `SIGBUS (0x7)` during `glfwDestroyWindow`, especially with xvfb-run

**Error:** Crash in `libnvidia-glcore.so` at application shutdown

**Cause:** NVIDIA OpenGL driver threading issue when running headless (xvfb-run)

**Fix:** Set environment variable before running:
```bash
export __GL_THREADED_OPTIMIZATIONS=0
xvfb-run -a sbt "run --optix ..."
```

**Prevention:**
- Already set in `.gitlab-ci.yml` for all CI jobs
- Already set in `.git_hooks/pre-push`
- Set in any local test scripts using xvfb-run

**Note:** This issue only affects headless rendering. Interactive display sessions typically don't crash.

## Package Issues

### Packaged app can't find optixjni library

**Expected behavior:** Known issue - packaged app tries loading from system path not bundled location. Doesn't affect code correctness. Only impacts distribution.

---

## User-Facing Common Issues

> This section covers runtime issues when using the rendered application.
> For build and JNI issues, see the sections above.

### Common Issues

#### "CUDA Error 718: Invalid Program Counter"

**Problem:** OptiX SDK version doesn't match NVIDIA driver version.

**Diagnosis:**
```bash
# Check driver version
nvidia-smi

# Check driver's OptiX version
strings /usr/lib/x86_64-linux-gnu/libnvoptix.so.* | grep "OptiX Version"

# Check SDK version used to build
grep "OptiX SDK:" optix-jni/target/native/x86_64-linux/build/CMakeCache.txt
```

**Solution:**
1. Driver 580.x+ requires OptiX SDK 9.0+
2. Driver 535-575.x requires OptiX SDK 8.0
3. Install matching SDK version
4. Clean and rebuild:
```bash
rm -rf optix-jni/target/native
sbt compile
```

For more details, see [INSTALLATION_FROM_SCRATCH.md](INSTALLATION_FROM_SCRATCH.md#check-optixdriver-compatibility).

#### "PTX File Not Found" or Solid Red Rendering

**Problem:** Compiled PTX shaders not found after `sbt clean`.

**Solution:**
```bash
# Rebuild project
sbt compile

# Or manually copy PTX
mkdir -p target/native/x86_64-linux/bin
cp optix-jni/target/classes/native/x86_64-linux/sphere_combined.ptx \
    target/native/x86_64-linux/bin/
```

#### Stale PTX After Shader Changes

**Problem:** Shader modifications don't take effect.

**Diagnosis:**
```bash
# Check PTX file timestamps
ls -lah */target/native/x86_64-linux/bin/sphere_combined.ptx \
    target/native/x86_64-linux/bin/sphere_combined.ptx
```

**Solution:**
```bash
# Remove stale PTX
rm -f target/native/x86_64-linux/bin/sphere_combined.ptx

# Or force-copy fresh PTX
cp optix-jni/target/native/x86_64-linux/bin/sphere_combined.ptx \
    target/native/x86_64-linux/bin/
```

#### "UnsatisfiedLinkError: no optixjni in java.library.path"

**Problem:** JNI library not found.

**Solution:**
```bash
# Check library exists
ls optix-jni/target/native/x86_64-linux/bin/liboptixjni.so

# Rebuild if missing
sbt "project optixJni" nativeCompile
```

#### SIGBUS Crash During Window Cleanup (Headless Mode)

**Problem:** JVM crashes with `SIGBUS` in `libnvidia-glcore.so` when using `xvfb-run`.

**Solution:**
```bash
# Set environment variable before running
export __GL_THREADED_OPTIMIZATIONS=0
xvfb-run sbt "run --optix ..."
```

This is already set in CI/CD configurations and git hooks.

#### Permission Errors After Docker Build

**Problem:** Files owned by root after Docker build.

**Solution:**
```bash
# Use pkexec (not sudo) per project guidelines
pkexec chown -R $USER:$USER optix-jni/target/
```

#### Out of Memory During Compilation

**Problem:** Compilation fails with OOM errors.

**Solution:**
```bash
# Limit sbt heap
sbt -J-Xmx4G compile

# Or add swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

For a complete troubleshooting guide, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Performance Tips

#### Optimizing Render Speed

**1. Choose the Right Geometry Type**
- For levels 0-4: Use surface subdivision (`sponge-surface`)
- For levels 5+: Use volume subdivision (`sponge-volume` with IAS)

**2. Reduce Quality for Previews**
```bash
# Fast preview (no AA, no shadows)
sbt "run --optix --objects 'type=sponge-surface:level=2'"

# Medium quality (AA only)
sbt "run --optix --objects 'type=sponge-surface:level=2' --antialiasing"

# Full quality (AA + shadows)
sbt "run --optix --objects 'type=sponge-surface:level=2' --antialiasing --shadows"
```

**3. Use LibGDX for Exploration**
```bash
# Quick interactive preview
sbt "run --sponge-type square-sponge --level 2"

# Then render final version with OptiX
sbt "run --optix --objects 'type=sponge-surface:level=2' --antialiasing --shadows"
```

**4. Limit Caustics Quality**
```bash
# Fast caustics preview
sbt "run --optix --objects 'type=sphere:ior=1.5' \
    --caustics --caustics-photons 50000 --caustics-iterations 5"

# Production caustics
sbt "run --optix --objects 'type=sphere:ior=1.5' \
    --caustics --caustics-photons 500000 --caustics-iterations 50"
```

**5. Optimize Antialiasing**
```bash
# Fastest AA
--antialiasing --aa-max-depth 1 --aa-threshold 0.2

# Balanced
--antialiasing --aa-max-depth 2 --aa-threshold 0.1

# Best quality
--antialiasing --aa-max-depth 4 --aa-threshold 0.05
```

**6. Headless Rendering for Batch Jobs**
```bash
export __GL_THREADED_OPTIMIZATIONS=0
xvfb-run sbt "run --optix ... --timeout 2.0"
```

#### Benchmarking

Check render statistics:
```bash
sbt "run --optix --objects 'type=sphere' --stats"
```

This displays ray counts, intersection tests, and timing information.

### Getting Help

#### Documentation Resources

- **Architecture**: [docs/arc42/README.md](arc42/README.md) - Full arc42 architecture documentation
- **Installation**: [docs/INSTALLATION_FROM_SCRATCH.md](INSTALLATION_FROM_SCRATCH.md) - Complete installation guide
- **Troubleshooting**: [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Detailed troubleshooting
- **Caustics**: [docs/caustics/CAUSTICS.md](caustics/CAUSTICS.md) - Caustics implementation details

#### Reporting Issues

If you encounter a bug or have a feature request:

1. Check existing issues: https://gitlab.com/lilacashes/menger/issues
2. Create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, GPU, driver version)
   - Relevant error messages

#### Contributing

Contributions are welcome! The project follows functional programming principles in Scala 3. See [AGENTS.md](../AGENTS.md) for code standards and development workflow.
