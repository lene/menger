# Installation from Scratch (menger)

The development stack — CUDA, OptiX, NVIDIA driver, Java, sbt — is installed once for the whole workspace; see the workspace repo's [docs/INSTALLATION_FROM_SCRATCH.md](../../docs/INSTALLATION_FROM_SCRATCH.md). This page covers cloning and building the **menger application** on top of that stack.

## Build and Test

### Clone Repository

```bash
git clone https://github.com/lene/menger.git   # GitHub is origin; GitLab is a read-only mirror
cd menger
```

### First Build (Downloads Dependencies)

The first build downloads Scala and dependencies and compiles native code. This takes 5-10 minutes:

```bash
sbt compile
```

**What happens:**
1. sbt downloads Scala 3 (version pinned in `build.sbt`, currently 3.8.3) and the dependencies —
   including the published artifacts `io.github.lene:optix-jni` (generic OptiX ray tracing,
   ships its own native library) and `io.github.lene:menger-common`. Neither is built here.
2. CMake configures the **menger-geometry** native build (`menger-geometry/src/main/native`)
3. nvcc compiles the Menger-specific 4D CUDA programs (→ `menger_4d.ptx`)
4. g++ compiles the video-decoding JNI bindings (libav) into `libmengergeometry.so`

To change optix-jni or menger-common, work in their own repos and bump the released version pin
in `build.sbt` — see the workspace `CLAUDE.md`.

### Run Tests

```bash
# Set __GL_THREADED_OPTIMIZATIONS=0 to prevent NVIDIA driver threading crashes under xvfb
export __GL_THREADED_OPTIMIZATIONS=0
xvfb-run sbt test
```

Expect roughly 2,000+ Scala tests, all passing. Before pushing, run the full gate instead —
`./.git_hooks/pre-push` (tests, scalafix, packaging, integration renders, coverage, memory
checks; ~8–10 minutes). See [TESTING.md](TESTING.md).

### Run Application

Every run needs `--objects` or `--scene`:

```bash
# Interactive window (requires a display)
sbt "run --objects type=sphere:size=1.5:material=glass --plane y:-2"

# Headless smoke test
export __GL_THREADED_OPTIMIZATIONS=0
xvfb-run -a sbt "run --objects type=sphere:size=1:material=glass --timeout 0.1"

# Render and save an image
xvfb-run -a sbt "run --objects type=sphere:size=1.5:material=glass --headless --save-name sphere.png"
```

## Troubleshooting

### CUDA Error 718 (OptiX Version Mismatch)

**Symptom:**
```
OptiX call failed: Invalid OptiX version (error code 718)
```

**Cause**: OptiX SDK version doesn't match driver version.

**Solution**:
1. Check driver version: `nvidia-smi`
2. Check driver's OptiX version: `strings /usr/lib/x86_64-linux-gnu/libnvoptix.so.* | grep "OptiX Version"`
3. Install matching SDK (9.0 for driver 580.x+, 8.0 for driver 535-575.x)
4. Clean rebuild: `sbt clean compile`

### cuda.h Not Found

**Symptom:**
```
fatal error: cuda.h: No such file or directory
```

**Cause**: `CUDA_HOME` not set or CUDA development files not installed.

**Solution**:
```bash
# Ensure CUDA toolkit (not just runtime) is installed
apt-get install cuda-toolkit-13-2

# Set environment variable
export CUDA_HOME=/usr/local/cuda

# Verify CUDA headers exist
ls $CUDA_HOME/include/cuda.h
```

### OptiX SDK Not Found

**Symptom:**
```
CMake Error: OptiX SDK not found. Set OPTIX_ROOT environment variable
```

**Cause**: `OPTIX_ROOT` not set or OptiX not installed.

**Solution**:
```bash
# Set environment variable
export OPTIX_ROOT=/usr/local/optix

# Verify OptiX headers exist
ls $OPTIX_ROOT/include/optix.h
```

### PTX File Not Found After sbt clean

**Symptom:**
```
RuntimeException: PTX file not found: menger_4d.ptx
```

**Cause**: `sbt clean` removes the compiled menger-geometry PTX, which is needed at runtime.

**Solution**: rebuild with `sbt compile`. (The generic OptiX shaders ship inside the
`optix-jni` jar and are not affected by `sbt clean`.)

### Permission Errors After Docker Build

**Symptom**: Files in `menger-geometry/target/` owned by root, can't delete locally.

**Cause**: Docker containers run as root.

**Solution**:
```bash
pkexec chown -R $USER:$USER menger-geometry/target/
```

### Out of Memory During Compilation

**Symptom**: g++ or sbt process killed, or "Out of memory" errors.

**Cause**: Insufficient RAM for parallel compilation.

**Solution**:
```bash
# Limit parallel compilation jobs
sbt -J-Xmx4G compile  # Limit sbt heap to 4GB

# Or increase system swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Additional Resources

- **Architecture**: workspace arc42 docs, `../docs/arc42/README.md` (in menger-toplevel)
- **Rendering physics / caustics**: [caustics/CAUSTICS.md](caustics/CAUSTICS.md)
- **CI runners**: `../infra/RUNNER_SETUP.md` (in menger-toplevel)
- **GPU development on AWS**: [guide/cloud.md](guide/cloud.md)
- **Troubleshooting**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for complete troubleshooting guide

## Quick Reference

```bash
# Environment variables (add to ~/.bashrc)
export CUDA_HOME=/usr/local/cuda
export OPTIX_ROOT=/usr/local/optix
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Common commands
sbt compile              # Build project
sbt test                 # Run all tests
sbt "run --objects type=sphere"   # Run application (needs --objects or --scene)
sbt clean                # Clean build artifacts
xvfb-run sbt test        # Headless test execution
```
