# Installation from Scratch (menger)

The development stack — CUDA, OptiX, NVIDIA driver, Java, sbt — is installed once for the whole workspace; see the workspace repo's [docs/INSTALLATION_FROM_SCRATCH.md](../../docs/INSTALLATION_FROM_SCRATCH.md). This page covers cloning and building the **menger application** on top of that stack.

## Build and Test

### Clone Repository

```bash
git clone https://gitlab.com/lilacashes/menger.git
cd menger
```

### First Build (Downloads Dependencies)

The first build will download Scala, dependencies, and compile native code. This takes 5-10 minutes:

```bash
# Compile project (includes C++/CUDA OptiX JNI)
sbt compile
```

**What happens:**
1. sbt downloads Scala 3.7.3 and project dependencies
2. CMake configures OptiX JNI build
3. nvcc compiles CUDA shaders (`.cu` → `.ptx`)
4. g++ compiles C++ JNI bindings
5. Creates `liboptixjni.so` shared library

### Run Tests

```bash
# Run all tests (menger + optix-jni)
# Use xvfb-run for headless execution
# Set __GL_THREADED_OPTIMIZATIONS=0 to prevent NVIDIA driver threading crashes
export __GL_THREADED_OPTIMIZATIONS=0
xvfb-run sbt test
```

**Expected output:**
- 16 C++ tests (OptiX context tests)
- 80+ Scala tests (rendering, physics, integration)
- All tests should pass

### Run Application

```bash
# Interactive mode (requires display)
sbt run

# Headless mode with OptiX sphere rendering
export __GL_THREADED_OPTIMIZATIONS=0
xvfb-run sbt "run --optix --objects type=sphere:size=1:material=glass --timeout 0.1"

# Render and save image
export __GL_THREADED_OPTIMIZATIONS=0
xvfb-run sbt "run --optix --objects type=sphere:size=1.5:material=glass --timeout 1.0 --save-name sphere.png"
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
4. Clean rebuild: `rm -rf optix-jni/target/native && sbt compile`

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
RuntimeException: PTX file not found: sphere_combined.ptx
```

**Cause**: `sbt clean` removes compiled PTX shaders but they're needed at runtime.

**Solution**:
```bash
# Rebuild project after clean
sbt compile

# Or manually copy PTX to expected location
mkdir -p target/native/x86_64-linux/bin
cp optix-jni/target/classes/native/x86_64-linux/sphere_combined.ptx \
    target/native/x86_64-linux/bin/
```

### Permission Errors After Docker Build

**Symptom**: Files in `optix-jni/target/` owned by root, can't delete locally.

**Cause**: Docker containers run as root.

**Solution**:
```bash
# Use pkexec instead of sudo (per CLAUDE.md)
pkexec chown -R $USER:$USER optix-jni/target/
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

- **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md) for code structure
- **OptiX Physics**: See [PHYSICS.md](PHYSICS.md) for rendering equations
- **CI/CD Setup**: See [CI_CD.md](CI_CD.md) for Docker image and runner configuration
- **GPU Development**: See [GPU_DEVELOPMENT.md](GPU_DEVELOPMENT.md) for AWS EC2 setup
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
sbt run                  # Run application
sbt clean                # Clean build artifacts
xvfb-run sbt test        # Headless test execution
```
