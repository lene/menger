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

### OptiX Validation Mode

**When to use:** Debugging SBT mismatches, payload size errors, or buffer
alignment issues that produce opaque CUDA error 718. Validation mode catches
these at the exact call site with a descriptive error.

**Usage:**
```bash
MENGER_OPTIX_VALIDATION=1 sbt run
MENGER_OPTIX_VALIDATION=1 menger-app --objects type=sphere
```

**Performance note:** Adds ~10-30% runtime overhead — debugging only, not production.

### CUDA error 35 ("driver version is insufficient for CUDA runtime version")

**Cause:** NVIDIA driver too old for the CUDA 13 runtime that the distributed native
libraries (`optix-jni` ≥0.1.3, `menger-geometry`) link against (`libcudart.so.13`).

**Symptom:** `cudaFree(0) failed: CUDA driver version is insufficient for CUDA runtime
version (35)` and `OptiXNotAvailableException: Failed to initialize OptiX renderer` — every
GPU render fails at startup, including a plain sphere.

**Diagnosis:**
```bash
nvidia-smi | grep -iE 'Driver Version|CUDA Version'   # need driver >= 580.65 / CUDA 13
ldd <jar-extracted>/native/x86_64-linux/liboptixjni.so | grep cudart   # libcudart.so.13
```

**Fix:** upgrade the NVIDIA driver to ≥580.65 (`pkexec apt install nvidia-driver-595`),
reboot. CUDA drivers are backward-compatible, so existing CUDA 12 builds keep working.

**Note:** the project standardized on CUDA 13 in Sprint 27 (arc42 §2 TC-4/TC-9). To stay on
an older driver, you would have to rebuild `optix-jni` against CUDA 12 and `publishLocal` it.

### CI GPU job "system failure": `open /run/nvidia-persistenced/socket: no such file or directory`

**Cause:** on a self-hosted GitLab GPU runner, the NVIDIA container toolkit mounts the
`nvidia-persistenced` socket; if the daemon is masked/inactive the job fails to start
(`OCI runtime create failed`) in seconds, before any test runs.

**Fix (on the runner host):**
```bash
pkexec systemctl unmask nvidia-persistenced
pkexec systemctl enable --now nvidia-persistenced
ls -la /run/nvidia-persistenced/socket   # confirm socket exists
```
Then retry the job (`glab ci retry <job-id>`).

### CI GPU job fails only, everything else green: `OPTIX_ERROR_UNKNOWN (7999)`

**Cause:** the self-hosted GPU runner's host NVIDIA driver was upgraded (often by the OS's
own automatic-update mechanism — PackageKit/GNOME-Software `aptdaemon`, or
`unattended-upgrades`) to a version the pinned OptiX runtime in the CI image
(`optix-cuda:$OPTIX_DOCKER_VERSION`) can't create a device context against. The container's
CUDA/OptiX version is fixed by the image tag; only the *host* driver floats, and
nvidia-container-toolkit injects whatever driver is currently loaded on the host into the
container.

**Symptom:** `Test:Full` / `Test:OptiXIntegration` (the two `tags: [nvidia]` jobs) fail while
every non-GPU job (Scalafix, CheckCoverage, SAST, code_quality) stays green. Job trace shows:
```
[OptiX][DEVICECTX]: Error initializing RTX library
[OptiXContext] Initialization failed: OptiX call 'optixDeviceContextCreate(...)' failed:
    OPTIX_ERROR_UNKNOWN (7999)
ERROR i.g.l.o.MengerRenderer - Failed to initialize OptiX renderer
```

**Diagnosis (on the runner host):**
```bash
nvidia-smi | grep -iE 'Driver Version|CUDA Version'
cat /proc/driver/nvidia/version                 # loaded kernel module
modinfo nvidia | grep ^version                  # on-disk module — must match the above
grep -iE 'nvidia-driver|nvidia-dkms' /var/log/apt/history.log | tail -20   # recent auto-upgrades
```
The job's own trace also prints the driver version the container saw (its `before_script`
runs `nvidia-smi`) — compare that against what the CI image's CUDA tag requires (e.g. driver
580.173.02 supports CUDA ≤13.0, but `OPTIX_DOCKER_VERSION` may require CUDA 13.2).

**Fix:** install a driver that supports the CI image's CUDA version, reboot, then confirm
`nvidia-smi` reports it before retrying CI:
```bash
pkexec apt install nvidia-driver-595   # or current minimum for OPTIX_DOCKER_VERSION's CUDA tag
pkexec reboot
# after reboot:
nvidia-smi                              # confirm new driver loaded
cd /path/to/menger
glab api --method POST "projects/:id/jobs/<failed-job-id>/retry"
```

**Prevention — freeze the driver once it works** (the root cause here was an *unrequested*
automatic upgrade, not a deliberate one):
```bash
dpkg -l | awk '/^ii/ && $2 ~ /(nvidia|cuda|libnvidia|libcuda)/ {print $2}' \
  | xargs pkexec apt-mark hold
pkexec apt-mark showhold   # verify
```
Also blacklist these packages from `unattended-upgrades` (belt-and-suspenders — `apt-mark
hold` alone is honored by apt/PackageKit/unattended-upgrades, but this makes the intent
explicit for the next reader): `/etc/apt/apt.conf.d/51-freeze-nvidia-cuda`:
```
Unattended-Upgrade::Package-Blacklist {
    "nvidia";
    "cuda";
    "libnvidia";
    "libcuda";
};
```
To intentionally upgrade later: `apt-mark unhold <pkgs>`, upgrade, **reboot**, then retry a
GPU CI job and confirm green **before** trusting the new driver — do not let a driver upgrade
on this host go unvalidated against CI.

**Second, independent cause of the same error signature:** even with a correct, matching
driver, `optixDeviceContextCreate` still fails with `OPTIX_ERROR_UNKNOWN (7999)` if
`libnvidia-rtcore.so` (OptiX's RTX core library) was never mounted into the container at all.
`nvidia-container-toolkit` gates this library behind the **`display`** (or `graphics`)
capability — **not** `compute`/`utility`. The job's `before_script` "create symlink for RTX
core library if needed" step is a no-op in this case: the glob
`libnvidia-rtcore.so.*` matches nothing because the file isn't there, so `test -f` is false
and the symlink is silently never created.

**Confirm this is the cause:**
```bash
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  $CI_REGISTRY_IMAGE/optix-cuda:$OPTIX_DOCKER_VERSION \
  bash -c 'ls /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so* || echo MISSING'
# add ",display" to NVIDIA_DRIVER_CAPABILITIES above and re-run — file should now be present
```

**Fix:** add `display` to `NVIDIA_DRIVER_CAPABILITIES` for every job that renders with OptiX
(`tags: [nvidia]`) — both the global default and each job's `variables:` override (GitLab CI
job-level `variables:` fully replaces the global block rather than merging, so every job that
redeclares `NVIDIA_VISIBLE_DEVICES` etc. must also redeclare this):
```yaml
NVIDIA_DRIVER_CAPABILITIES: "compute,utility,display"
```
`display` (not the broader `graphics`) mounts the NVIDIA core libs OptiX needs (including
`libnvidia-rtcore.so`) without also pulling in windowing-system EGL bridge libraries
(`libnvidia-egl-xcb`/`-wayland`/`-xlib`) that caused a host-side dpkg conflict in an earlier
driver upgrade — verified empirically by comparing the mounted library set under `display`
vs `graphics` vs `compute,utility` alone.

**Why this can appear "new" after a driver upgrade:** the capability-to-library classification
lives in `nvidia-container-toolkit`, not the driver package. If a working `compute,utility`-only
setup (as this repo's `.gitlab-ci.yml` used) suddenly regresses to this error after a driver or
toolkit upgrade, check `nvidia-container-cli list` output and compare against what the running
container actually receives (`docker run --gpus all ... bash -c 'ls .../libnvidia-rtcore.so*'`)
rather than assuming the driver version itself is still the problem.

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

For performance tips and further help, see [Performance Tips](#performance-tips) and [Getting Help](#getting-help) below.

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

#### Shader Execution Reordering (SER)

**Ada Lovelace+ GPUs (RTX 40xx+):** Enable `MENGER_OPTIX_SER=1` to use OptiX
shader execution reordering, which improves SIMT coherence for divergent rays
(e.g., sponges with mixed materials). Enabling SER requires a pipeline rebuild
on next render — expect 5-20% frame-time improvement on divergent scenes.

```bash
MENGER_OPTIX_SER=1 menger-app --objects type=sponge-volume:level=4
```

Disabled by default pending benchmarking. Use `scripts/benchmark.sh` to
compare on/off timing on your GPU.

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
