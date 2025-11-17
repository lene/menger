# Installation from Scratch

This guide walks through installing the Menger sponge renderer from a fresh Ubuntu/Debian system, including all dependencies (CUDA, OptiX, Java, sbt). This is useful for setting up development environments or understanding the full dependency stack.

For most users, the pre-built Docker image is recommended (see [CI_CD.md](CI_CD.md)).

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Install System Dependencies](#install-system-dependencies)
3. [Install CUDA Toolkit](#install-cuda-toolkit)
4. [Install OptiX SDK](#install-optix-sdk)
5. [Install Java and sbt](#install-java-and-sbt)
6. [Build and Test](#build-and-test)
7. [Troubleshooting](#troubleshooting)

## System Requirements

- **OS**: Ubuntu 22.04+ or Debian stable/testing
- **GPU**: NVIDIA GPU with RTX support (for OptiX ray tracing)
- **Driver**: NVIDIA driver 535+ (for OptiX 9.0)
- **Disk**: ~15 GB for CUDA toolkit and dependencies
- **RAM**: 8 GB minimum, 16 GB recommended

## Install System Dependencies

```bash
# Update package lists
apt-get -y update
apt-get -y upgrade

# Install build tools and X11 utilities
apt-get -y install \
    cmake \
    g++ \
    curl \
    mesa-utils \
    x11-xserver-utils \
    xvfb
```

**What these do:**
- `cmake` - Build system for C++/CUDA code
- `g++` - C++ compiler
- `curl` - Download tools
- `mesa-utils`, `x11-xserver-utils` - OpenGL/X11 utilities
- `xvfb` - Virtual framebuffer for headless rendering

## Install CUDA Toolkit

CUDA Toolkit 12.8 is required for OptiX 9.0 compatibility.

### Add NVIDIA Repository

```bash
# Download and install CUDA repository keyring
curl -o cuda-keyring_1.1-1_all.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get -y update
```

### Install CUDA Toolkit 12.8

```bash
# This takes ~10 minutes and downloads ~3 GB
apt-get -y install cuda-toolkit-12-8
```

### Set Environment Variables

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Add these to your `~/.bashrc` or `~/.profile` to persist across sessions:

```bash
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Verify CUDA Installation

```bash
nvcc --version
# Should output: Cuda compilation tools, release 12.8
```

## Install OptiX SDK

OptiX SDK 9.0 is required for driver 580.x+ compatibility.

### Check OptiX/Driver Compatibility

```bash
# Check driver version
nvidia-smi

# Check installed OptiX version in driver
strings /usr/lib/x86_64-linux-gnu/libnvoptix.so.* | grep "OptiX Version"
```

**Compatibility Matrix:**
- Driver 580.x+ → OptiX SDK 9.0+
- Driver 535-575.x → OptiX SDK 8.0

### Download OptiX SDK

**Option 1: Download from NVIDIA (Requires Login)**

1. Go to https://developer.nvidia.com/designworks/optix/downloads
2. Login with NVIDIA Developer account (free)
3. Download `NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh`

**Option 2: Use Project Package Registry (CI/Team Members)**

```bash
# Requires GitLab access token with read_api scope
curl --header "PRIVATE-TOKEN: YOUR_TOKEN" \
    -o NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh \
    "https://gitlab.com/api/v4/projects/53243565/packages/generic/optix-sdk/9.0.0/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh"
```

### Install OptiX SDK

```bash
# Make installer executable
chmod +x NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh

# Install to standard location
mkdir -p /usr/local/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64
sh NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh \
    --skip-license \
    --prefix=/usr/local/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64

# Create convenience symlink
ln -s /usr/local/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64 /usr/local/optix
```

### Set OptiX Environment Variable

```bash
export OPTIX_ROOT=/usr/local/optix
```

Add to `~/.bashrc` to persist:

```bash
echo 'export OPTIX_ROOT=/usr/local/optix' >> ~/.bashrc
source ~/.bashrc
```

## Install Java and sbt

### Install Java 17+

```bash
# Install OpenJDK 17 (minimum version)
apt-get -y install openjdk-17-jdk

# Verify installation
java -version
# Should show: openjdk version "17.x.x" or higher
```

**Note**: Java 21 or 25 is recommended for better performance, but 17 is the minimum supported version.

### Install sbt (Scala Build Tool)

```bash
# Add sbt repository
echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" \
    > /etc/apt/sources.list.d/sbt.list
echo "deb https://repo.scala-sbt.org/scalasbt/debian /" \
    > /etc/apt/sources.list.d/sbt_old.list

# Add repository key
curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" \
    | apt-key add

# Update and install sbt
apt-get -y update
apt-get -y install sbt

# Verify installation
sbt --version
# Should show: sbt version 1.x.x
```

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
xvfb-run sbt "run --optix --sponge-type sphere --timeout 0.1 --radius 0.5"

# Render and save image
xvfb-run sbt "run --optix --sponge-type sphere --timeout 1.0 --save-name sphere.png"
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
apt-get install cuda-toolkit-12-8

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

## Automated Validation

This installation procedure is automatically validated by the `Test:SbtImage` CI job, which runs weekly to ensure these instructions remain accurate. See `.gitlab-ci.yml` for the automated version.
