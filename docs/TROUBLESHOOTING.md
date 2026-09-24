← [User Guide Index](USER_GUIDE.md)

# Troubleshooting

**Where the native code lives** (it matters for most fixes below): the generic OptiX layer —
`liboptixjni.so` and `optix_shaders.ptx` — ships *inside* the `io.github.lene:optix-jni` jar
from Maven Central and is not built in this repo. This repo builds only `menger-geometry`:
`menger_4d.ptx` (the 4D fractal programs) and `libmengergeometry.so` (video decoding), under
`menger-geometry/target/native/x86_64-linux/bin/`. Everything is compiled by `sbt compile`.

Quick index:

| Symptom | Section |
|---------|---------|
| `Error: SceneConfig must provide objectSpecs` | [Nothing to render](#nothing-to-render) |
| `Error: Unknown option 'optix'` (or `object`, `radius`, …) | [Removed options](#removed-options) |
| CUDA error 35 at startup | [CUDA error 35](#cuda-error-35-driver-version-is-insufficient-for-cuda-runtime-version) |
| CUDA error 718 | [CUDA error 718](#cuda-error-718-invalid-program-counter) |
| CUDA error 719 / out of memory in tests | [GPU busy](#gpu-busy--oom-from-a-concurrent-process-cuda-error-719-out-of-memory) |
| `menger_4d.ptx not found` | [Missing PTX](#menger_4dptx-not-found-after-sbt-clean) |
| `failed to load` (native library) | [Native library](#native-library-failed-to-load--unsatisfiedlinkerror) |
| JVM crash in `libnvidia-glcore.so` under `xvfb-run` | [SIGBUS](#sigbus-crash-in-libnvidia-glcoreso-under-xvfb-run) |
| `render lock already held: <path>` / second window refused | [Render lock](#second-interactive-window-refused) |
| `.scala` scene stopped compiling after upgrading to 0.9.0 | [Restricted classpath](#scene-file-no-longer-compiles-090) |

## Usage Errors

### Nothing to render

**Symptom:** `Error: SceneConfig must provide objectSpecs` (often after a bare `sbt run`).

**Cause:** every run needs `--objects` or `--scene`; there is no default scene.

**Fix:** `sbt "run --objects type=sphere"` or `sbt "run --scene glass-sphere"`.

### Removed options

**Symptom:** `Error: Unknown option 'optix'` (or `object`, `radius`, `scale`, `center`, `ior`).

**Cause:** OptiX is the only renderer since the LibGDX renderer was removed, so `--optix` is gone;
the single-object flags were replaced by `--objects`.

**Fix:** drop `--optix`; write `--objects type=sphere:size=1.5:pos=0,0,0:ior=1.5` instead of the
single-object flags. Some older flags are still *accepted but ignored* (`--sponge-type`,
`--lines`, `--color`, …) — see the [User Guide](guide/user-guide.md#legacy-flags-that-currently-do-nothing).

### Second interactive window refused

**Symptom:** starting a second interactive window fails immediately with
`render lock already held: <path>`.

**Cause:** at most one interactive render session may be active (lock file, `--render-lock-path`).
This is deliberate — a second request is refused, never queued. Headless renders are not affected.

**Fix:** close the other window, or render headless (`--headless --save-name out.png`). If no
window is open, a crashed process may still hold the lock only until it exits; the OS releases
the lock when the process ends.

### Scene file no longer compiles (0.9.0)

**Symptom:** a `.scala` scene loaded with `--scene file.scala` that worked in 0.8.x fails to
compile with missing-package errors.

**Cause:** since 0.9.0 scene files compile against a restricted classpath: the Scala library,
menger-common, scala-logging, and `menger.dsl`, `menger.objects`, `menger.video`. Imports of
LibGDX, `io.github.lene.optix`, `upickle`, or menger's `engines`/`tools`/`cli`/`input` packages
are rejected.

**Fix:** use only the DSL (`import menger.dsl.*`). See the
[DSL reference](guide/dsl-reference.md#basic-dsl-structure).

## GPU, Driver and Build Issues

### CUDA error 35 ("driver version is insufficient for CUDA runtime version")

**Cause:** NVIDIA driver too old for the CUDA 13 runtime that the native libraries
(`optix-jni` ≥0.1.3, `menger-geometry`) link against (`libcudart.so.13`).

**Symptom:** `cudaFree(0) failed: CUDA driver version is insufficient for CUDA runtime
version (35)` and `OptiXNotAvailableException: Failed to initialize OptiX renderer` — every
GPU render fails at startup, including a plain sphere.

**Diagnosis:**
```bash
nvidia-smi | grep -iE 'Driver Version|CUDA Version'   # need driver >= 580.65 / CUDA 13
```

**Fix:** upgrade the NVIDIA driver to ≥580.65 (e.g. `pkexec apt install nvidia-driver-595`) and
reboot. The project standardized on CUDA 13 in Sprint 27; staying on an older driver would mean
rebuilding `optix-jni` against CUDA 12 and `publishLocal`-ing it.

### CUDA error 718 ("invalid program counter")

**Cause:** OptiX SDK / driver mismatch (OptiX has a strict ABI — the SDK a PTX was built against
must be supported by the driver's OptiX runtime), or a pipeline/SBT bug.

**Diagnosis:**
```bash
nvidia-smi                                                             # driver version
strings /usr/lib/x86_64-linux-gnu/libnvoptix.so.* | grep "OptiX Version"   # driver's OptiX
grep "OptiX" menger-geometry/target/native/x86_64-linux/build/CMakeCache.txt  # SDK used here
```

**Fix:** install OptiX SDK 9.0 (the project's target; driver ≥580.65 supports it), point
`OPTIX_ROOT` at it, then `sbt clean compile`. If the published `optix-jni` jar itself fails with
718 on your driver, that is a driver problem — upgrade the driver. If the error only appears
with a particular scene, use validation mode (below).

### OptiX validation mode

**When to use:** SBT mismatches, payload-size errors or buffer-alignment problems that surface as
an opaque CUDA error 718. Validation mode reports them at the exact call site.

```bash
MENGER_OPTIX_VALIDATION=1 sbt "run --objects type=sphere"
MENGER_OPTIX_VALIDATION=1 menger-app --objects type=sphere
```

Adds ~10-30% runtime overhead — debugging only.

### GPU busy / OOM from a concurrent process (CUDA error 719, out of memory)

A foreign process (video encoding, another render, a second Menger instance) holding GPU memory
or compute time can make `memcheck`/`integration` fail with CUDA error 719 or out-of-memory
errors unrelated to the code under test. The pre-push hook's `gpu-preflight.sh` (Sprint 36 D1)
detects this: free VRAM below ~2 GiB or any listed compute process triggers a short retry window,
then a labeled `SKIP (env: GPU busy — <process>)` locally, or an `ENV-UNSUITABLE` CI failure
(rerun with `gh run rerun <run-id> --failed` once the GPU is free). Manual check:
`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.

### `menger_4d.ptx not found` after `sbt clean`

**Symptom:** `menger_4d.ptx not found on classpath (/native/x86_64-linux/menger_4d.ptx) or in
[menger-geometry/target/native/x86_64-linux/bin/menger_4d.ptx, …]`, typically as soon as a 4D
fractal (`menger4d`, `sierpinski4d`, `hexadecachoron4d`) is rendered.

**Cause:** `sbt clean` deleted menger-geometry's native build output. The PTX is looked up on the
classpath first (packaged app), then in the sbt build directory (`sbt run` / tests).

**Fix:** `sbt compile` (rebuilds it). Run `sbt` from the repo root — the fallback paths are
relative to the working directory.

### Native library failed to load / `UnsatisfiedLinkError`

**Symptom:** `OptiXNotAvailableException: Menger native library (mengergeometry) failed to load`,
or an `UnsatisfiedLinkError` for `optixjni`.

**Diagnosis and fix:**
- `mengergeometry`: check `menger-geometry/target/native/x86_64-linux/bin/libmengergeometry.so`
  exists; if not, `sbt compile`. `sbt run` sets `java.library.path` to that directory
  (`build.sbt`); the packaged app bundles it.
- `optixjni`: it is loaded from the optix-jni jar on the classpath. A failure there usually means
  the CUDA runtime isn't found: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`,
  and check the driver (error 35 above).

### "OptiX headers not found" / "cannot find -lcuda" when building

- Set `OPTIX_ROOT` to the OptiX SDK path (e.g. `/usr/local/optix`); CMake prefers the highest
  installed SDK version.
- Check CUDA 13.x is installed and on `PATH`: `nvcc --version`; set `CUDA_HOME=/usr/local/cuda`.
- Full setup: the workspace [installation guide](../../docs/INSTALLATION_FROM_SCRATCH.md).

### CMake "Ignoring extra path" warnings

Handled by the custom `CMakeWithoutVersionBug` build tool (`project/CMakeWithoutVersionBug.scala`),
which works around sbt-jni's version passing. No action needed.

### Root-owned files after Docker builds (`AccessDeniedException`, CMake cache mismatch)

**Cause:** a container ran as root and left root-owned files, or a CMake cache created at a
different path inside a container.

**Fix:**
```bash
pkexec chown -R $USER:$USER menger-geometry/target/
rm -rf menger-geometry/target/native     # only if the CMake cache path mismatches
```
**Prevention:** run containers with `--user $(id -u):$(id -g)`.

### SIGBUS crash in `libnvidia-glcore.so` under `xvfb-run`

**Symptom:** JVM crashes with `SIGBUS (0x7)` during `glfwDestroyWindow` at shutdown, especially
under `xvfb-run`.

**Cause:** NVIDIA OpenGL driver threading issue in headless X sessions.

**Fix:** always set the variable before headless runs (the pre-push hook and CI already do):
```bash
export __GL_THREADED_OPTIMIZATIONS=0
xvfb-run -a sbt "run --objects type=sphere --headless --save-name out.png"
```

### Out of memory during compilation

```bash
sbt -J-Xmx4G compile        # limit sbt heap
```
or add swap space.

## CI Issues (self-hosted GitHub Actions runners)

CI runs on GitHub Actions; GPU jobs run bare on self-hosted runners labelled `nvidia` (see the
workspace `infra/ci-runners/README.md`). Retry failed jobs with `gh run rerun <run-id> --failed`.

### Only GPU jobs fail with `OPTIX_ERROR_UNKNOWN (7999)` after a driver change

**Cause:** the runner host's NVIDIA driver was upgraded — often by the OS's automatic updates
(PackageKit / `unattended-upgrades`) — and the loaded kernel module no longer matches the
installed driver, or the new driver doesn't support the CUDA version in use.

**Symptom:** GPU jobs fail with
```
[OptiX][DEVICECTX]: Error initializing RTX library
[OptiXContext] Initialization failed: ... optixDeviceContextCreate(...) failed: OPTIX_ERROR_UNKNOWN (7999)
```
while every non-GPU job stays green.

**Diagnosis (on the runner host):**
```bash
nvidia-smi | grep -iE 'Driver Version|CUDA Version'
cat /proc/driver/nvidia/version                 # loaded kernel module
modinfo nvidia | grep ^version                  # on-disk module — must match the above
grep -iE 'nvidia-driver|nvidia-dkms' /var/log/apt/history.log | tail -20
```

**Fix:** install a driver supporting CUDA 13.x, **reboot**, confirm with `nvidia-smi`, then
`gh run rerun <run-id> --failed`.

**Prevention — freeze the driver once it works:**
```bash
dpkg -l | awk '/^ii/ && $2 ~ /(nvidia|cuda|libnvidia|libcuda)/ {print $2}' \
  | xargs pkexec apt-mark hold
pkexec apt-mark showhold   # verify
```
and blacklist the packages from `unattended-upgrades` in
`/etc/apt/apt.conf.d/51-freeze-nvidia-cuda`:
```
Unattended-Upgrade::Package-Blacklist {
    "nvidia";
    "cuda";
    "libnvidia";
    "libcuda";
};
```
To upgrade deliberately later: `apt-mark unhold <pkgs>`, upgrade, reboot, and confirm a GPU CI
run is green before trusting the new driver.

**Historical note (container-based CI):** under the retired GitLab/Docker CI the same error also
appeared when `libnvidia-rtcore.so` was not mounted into the container — `nvidia-container-toolkit`
only mounts it with the `display` (or `graphics`) capability, so jobs needed
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,display`. Keep this in mind for any containerized
GPU run (e.g. the scene-validator sandbox if it is ever given GPU access).

### Runner never picks up jobs

GitHub deletes runner registrations that stay disconnected too long. If the runner log says
*"The runner registration has been deleted from the server"*, re-register it — see the workspace
`infra/ci-runners/README.md`.

---

## Performance Tips

### Optimizing Render Speed

**1. Choose the Right Geometry Type**
- For levels 0-4: Use surface subdivision (`sponge-surface`)
- For levels 5+: Use volume subdivision (`sponge-volume` with IAS)
- For very deep levels (6+): `sponge-recursive-ias` (constant memory per level)

**2. Reduce Quality for Previews**
```bash
# Fast preview (no AA, no shadows)
sbt "run --objects 'type=sponge-surface:level=2'"

# Medium quality (AA only)
sbt "run --objects 'type=sponge-surface:level=2' --antialiasing"

# Full quality (AA + shadows)
sbt "run --objects 'type=sponge-surface:level=2' --antialiasing --shadows --plane y:-2"
```

**3. Explore Cheaply, Then Render**
```bash
# Explore interactively at a low level with no extra passes
sbt "run --objects 'type=sponge-surface:level=1'"

# Then render the final version headless with everything on
sbt "run --objects 'type=sponge-surface:level=3' --antialiasing --shadows \
    --plane y:-2 --headless --save-name final.png"
```

For animated DSL scenes, `--preview` lets you scrub through `t` before rendering all frames.

**4. Limit Caustics Quality**
```bash
# Fast caustics preview
sbt "run --objects 'type=sphere:ior=1.5' --plane y:-2 \
    --caustics --caustics-photons 50000 --caustics-iterations 5"

# Production caustics
sbt "run --objects 'type=sphere:ior=1.5' --plane y:-2 \
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
xvfb-run -a sbt "run --objects 'type=sponge-volume:level=3' --headless --save-name out.png"
```

### Benchmarking

```bash
sbt "run --objects 'type=sphere' --stats"                        # print statistics
sbt "run --objects 'type=sphere' --headless --save-name s.png --stats-json stats.json"
```

`--stats` prints frame time, ms per million rays and ray counts by type (primary, reflected,
refracted, shadow, AA). `scripts/benchmark.sh` compares frame times against
`scripts/perf-baseline.json`.

### Shader Execution Reordering (SER)

**Ada Lovelace+ GPUs (RTX 40xx+):** `MENGER_OPTIX_SER=1` enables OptiX shader execution
reordering, which improves SIMT coherence for divergent rays (e.g. sponges with mixed materials).
It triggers a pipeline rebuild on the next render — expect 5-20% frame-time improvement on
divergent scenes.

```bash
MENGER_OPTIX_SER=1 menger-app --objects type=sponge-volume:level=4
```

Disabled by default pending benchmarking. Use `scripts/benchmark.sh` to compare on/off timing on
your GPU.

## Getting Help

### Documentation Resources

- **Architecture**: [arc42 (workspace repo)](../../docs/arc42/README.md)
- **Installation**: [INSTALLATION_FROM_SCRATCH.md](INSTALLATION_FROM_SCRATCH.md) (and the
  workspace-level [stack setup](../../docs/INSTALLATION_FROM_SCRATCH.md))
- **Usage**: [User Guide](USER_GUIDE.md)
- **Caustics**: [caustics/CAUSTICS.md](caustics/CAUSTICS.md)

### Reporting Issues

1. Check existing issues: https://github.com/lene/menger/issues
2. Create a new issue with:
   - Clear description of the problem
   - Steps to reproduce (the exact `menger-app` / `sbt "run ..."` command)
   - Expected vs actual behavior
   - Environment details (OS, GPU, `nvidia-smi` driver version, menger version from `--version`)
   - Relevant error messages

### Contributing

Contributions are welcome! The project follows functional programming principles in Scala 3. See
[AGENTS.md](../AGENTS.md) for code standards and development workflow.
