# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ CRITICAL: Alpha Channel Convention

**STANDARD GRAPHICS ALPHA CONVENTION (DO NOT FORGET THIS):**
- **alpha = 0.0** → **FULLY TRANSPARENT** (fully transmissive, no opacity, no absorption)
- **alpha = 1.0** → **FULLY OPAQUE** (no transmission, full opacity, maximum absorption)

This is the standard graphics convention used throughout the codebase. This applies to:
- All OptiX shader code (`sphere_combined.cu`)
- All Beer-Lambert absorption calculations
- All color parameters in Scala code
- All LibGDX Color objects (Color.a field)
- All test expectations

**Never confuse this.** If you see alpha described as "absorption intensity where 0.0 = fully absorbing", that is **WRONG** and must be corrected.

## Active Development

**Current Work:** Phase 4 - OptiX Integration and Testing (Issue #45) - ✅ **COMPLETE**
- Branch: `45-phase-4-optix-integration-and-testing`
- See detailed plan: [PHASE_4_PLAN.md](./PHASE_4_PLAN.md)

**Status:**
- ✅ All OptiX integration complete (29 tests passing)
- ✅ Memory leak verification complete (compute-sanitizer + Valgrind: 0 leaks)
- ✅ Visual validation complete (optix_test_output.ppm)
- ✅ Valgrind host memory testing added to CI pipeline (Test:Valgrind job)
- ✅ compute-sanitizer GPU memory testing added to CI pipeline (Test:ComputeSanitizer job)
- ✅ Both tools integrated into pre-push hook for local validation
- **Next:** Verify CI pipeline passes, then close GitLab issue #45

## Project Overview

Menger is a Scala 3 application that renders fractals in 3D and 4D using LibGDX for graphics. It
generates Menger sponges (3D), tesseract sponges (4D), and their variants through recursive
subdivision algorithms. The project supports interactive exploration, animations, and fractional
iteration levels with alpha blending.

The project consists of two main components:
- **Core Scala application** (`menger`): LibGDX-based fractal renderer
- **OptiX JNI bindings** (`optix-jni`): Native CUDA/OptiX integration for GPU-accelerated ray
  tracing

## Build Requirements

**OptiX JNI is now a required dependency.** Building the project requires:

- CUDA Toolkit 12.0+
- NVIDIA OptiX SDK 9.0+ (8.0 is NOT compatible with modern drivers)
- CMake 3.18+
- C++ compiler with C++17 support

If CUDA or OptiX SDK are not installed, the build will fail with a clear error message.

**Runtime flexibility:** While CUDA/OptiX are required at build time, the compiled application
can run on systems without NVIDIA GPUs. The `OptiXRenderer.isAvailable` method performs runtime
checks for GPU hardware and drivers, allowing graceful degradation when GPU is unavailable.

**CRITICAL: OptiX SDK Version Compatibility**

OptiX SDK version MUST match the OptiX runtime version in your NVIDIA driver:
- **Driver 580.x+ includes OptiX 9.0** → Requires OptiX SDK 9.0+
- **Driver 535.x-575.x includes OptiX 8.0** → Requires OptiX SDK 8.0
- Mismatches cause CUDA error 718 ("invalid program counter") during GPU execution
- The build system auto-detects and prefers the highest SDK version installed
- Check driver's OptiX version: `strings /usr/lib/x86_64-linux-gnu/libnvoptix.so.* | grep "OptiX Version"`

## Development Commands

**Note**: Commands in this document use `sudo` for elevated privileges where needed. If your system doesn't have `sudo` configured (e.g., some Ubuntu configurations), use `pkexec` instead (e.g., `pkexec systemctl restart docker` instead of `sudo systemctl restart docker`).

### Basic Development

```bash
# Compile (includes native C++/CUDA compilation)
sbt compile

# Run tests (includes OptiX JNI tests if GPU available)
sbt test --warn

# Run specific test
sbt "testOnly menger.objects.CubeTest"

# Run application
sbt run

# Start Scala REPL
sbt console

# Check code quality
sbt "scalafix --check"
```

### OptiX JNI Development

```bash
# Compile native code only
sbt "project optixJni" nativeCompile

# Clean and rebuild native code
rm -rf optix-jni/target/native && sbt "project optixJni" compile

# Run OptiX JNI tests
sbt "project optixJni" test

# View generated PTX files (CUDA kernels)
ls optix-jni/target/native/x86_64-linux/bin/*.ptx

# Package application
sbt "Universal / packageBin"
```

**Build output summary:**

```
========================================
OptiX JNI Build Configuration:
  CUDA Version:   12.8.0
  OptiX SDK:      /usr/local/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64
========================================
```

### Packaging and Integration Tests

```bash
# Package application (creates ZIP in target/universal/)
sbt "Universal / packageBin" --warn

# Extract and test packaged application
VERSION=$(grep 'version :=' build.sbt | cut -d '"' -f 2)
unzip -oq ./target/universal/menger-${VERSION}.zip

# Integration tests (requires xvfb for headless rendering)
xvfb-run -a ./menger-${VERSION}/bin/menger --level 2 --timeout 0.1 --sponge-type cube --lines
xvfb-run -a ./menger-${VERSION}/bin/menger --level 4 --timeout 0.1 --sponge-type square
xvfb-run -a ./menger-${VERSION}/bin/menger --level 2 --timeout 0.1 --sponge-type tesseract-sponge
xvfb-run -a ./menger-${VERSION}/bin/menger --level 1.5 --timeout 0.1 --sponge-type tesseract-sponge-2

# Test animation rendering
rm -f test_image*.png
xvfb-run -a ./menger-${VERSION}/bin/menger --level 1 --sponge-type tesseract-sponge-2 --animate frames=5:rot-y-w=0-90:rot-y=0-30 --save-name test_image%d.png
ls test_image*.png 2>&1
```

### Expected Test Results

- **Compilation**: Clean compile with no errors (includes native C++/CUDA compilation)
- **Tests**: All 762+ tests pass (Scala: 750+, OptiX JNI: 15)
- **Native artifacts**: Shared library (`liboptixjni.so`) and PTX file generated
- **Packaging**: Creates ~22MB ZIP file
- **Integration**: All sponge types render successfully with expected face counts
- **Animation**: Generates 5 PNG files (800x600 RGBA)

**Note**: OptiX JNI tests require GPU hardware and proper runner configuration (see CI/CD Configuration section).

## Code Architecture

### Core Components

**Main.scala**: Application entry point. Creates LibGDX configuration and instantiates either
`InteractiveMengerEngine` (for interactive mode) or `AnimatedMengerEngine` (for animation mode)
based on CLI options.

**MengerEngine**: Abstract base class for rendering engines. Contains the `generateObject()`
factory method that creates geometry instances based on sponge type string. Supports overlay mode
for rendering transparent faces with wireframe overlays.

**Geometry trait** (`menger.objects.Geometry`): Base trait for all renderable objects. Implements
the Observer pattern for responding to rotation/projection parameter changes. All geometric objects
extend this trait and implement `getModel: List[ModelInstance]`.

### Fractal Generation Strategy

The codebase uses **surface subdivision** rather than volume subdivision for efficiency:

- **Volume approach** (O(20^n) for 3D): Subdivide cube into 27 smaller cubes, remove 7 to create
  holes
- **Surface approach** (O(12^n) for 3D): Subdivide each square face into 9 smaller squares, remove
  center square and add 4 squares defining the hole

Surface subdivision generates only the outer surface with no internal faces, significantly reducing
computational complexity.

### Object Hierarchy

```
menger.objects/
  - Geometry (trait)          # Base trait for all renderable objects
  - Square, Cube              # Basic 2D/3D primitives
  - SpongeBySurface           # 3D Menger sponge (12 faces per face)
  - SpongeByVolume            # 3D Menger sponge (20 cubes per cube)
  - FractionalLevelSponge     # Trait for fractional level support with alpha blending
  - Composite                 # Combines multiple geometries with overlay support

  higher_d/
    - Tesseract               # 4D hypercube
    - TesseractSponge         # 4D sponge (48 tesseracts per tesseract)
    - TesseractSponge2        # 4D sponge (16 faces per face)
    - RotatedProjection       # Wraps 4D objects with rotation and projection to 3D
    - FractionalRotatedProjection  # Adds fractional level support to 4D objects
    - Mesh4D                  # Represents 4D mesh data structure
```

### OptiX JNI Architecture

The `optix-jni` subproject provides JNI bindings to NVIDIA OptiX for GPU-accelerated ray tracing using a **two-layer architecture**:

#### Low-Level Layer: OptiXContext
- Pure OptiX API wrapper with no scene state
- Stateless where possible (only holds OptiX device context)
- Explicit resource management (create/destroy pairs for all resources)
- Direct 1:1 mapping to OptiX operations
- Unit tested with Google Test (16 C++ tests)

**Files:**
- `include/OptiXContext.h` / `OptiXContext.cpp`

**Key methods:**
- `initialize()` / `destroy()` - OptiX device context lifecycle
- `createModuleFromPTX()` / `destroyModule()` - Shader compilation
- `createRaygenProgramGroup()` / `createMissProgramGroup()` / `createHitgroupProgramGroup()` - Program group creation
- `createPipeline()` / `destroyPipeline()` - Pipeline assembly
- `buildCustomPrimitiveGAS()` / `destroyGAS()` - Geometry acceleration structure
- `createRaygenSBTRecord()` / `createMissSBTRecord()` / `createHitgroupSBTRecord()` - Shader binding table records
- `launch()` - OptiX kernel execution

#### High-Level Layer: OptiXWrapper
- Scene state management (sphere position/radius, camera, light, plane, material properties)
- Convenience methods for scene setup
- Performance optimization (scene data in Params instead of SBT for faster parameter updates)
- Uses OptiXContext for all OptiX operations (composition, not inheritance)

**Files:**
- `include/OptiXWrapper.h` / `OptiXWrapper.cpp`
- `include/OptiXData.h` - Shared data structures (Params, SBT records)

**Key methods:**
- `setSphere()`, `setSphereColor()`, `setIOR()`, `setScale()` - Scene configuration
- `setCamera()`, `setLight()`, `setPlane()` - Environment setup
- `render()` - High-level rendering (builds pipeline if needed, launches OptiX, returns RGBA image)

#### JNI Interface
- Binds Scala `OptiXRenderer` to C++ `OptiXWrapper`
- Per-instance native handles (supports multiple renderer instances)
- Error propagation via return codes
- Functional-style library loading with Try monad

**Files:**
- `JNIBindings.cpp` - C++ JNI native method implementations
- `menger/optix/OptiXRenderer.scala` - Scala API with functional library loading

#### Shaders (CUDA)
- Single combined shader file (`sphere_combined.cu`) compiled to PTX
- Ray generation, miss, closest hit, and custom sphere intersection programs
- Reads dynamic scene data from Params struct (not SBT) for performance

**Files:**
- `shaders/sphere_combined.cu` - All shaders in one file
- `target/native/x86_64-linux/bin/sphere_combined.ptx` - Compiled PTX

#### Directory Structure
```
optix-jni/src/main/
  native/
    CMakeLists.txt          # CMake build configuration
    OptiXContext.cpp        # Low-level OptiX wrapper
    OptiXWrapper.cpp        # High-level scene renderer
    JNIBindings.cpp         # JNI interface
    include/
      OptiXContext.h
      OptiXWrapper.h
      OptiXData.h           # Shared data structures
      OptiXConstants.h      # Magic number constants
    shaders/
      sphere_combined.cu    # Combined CUDA shaders
    tests/
      OptiXContextTest.cpp  # Google Test unit tests (16 tests)

  scala/menger/optix/
    OptiXRenderer.scala     # Main Scala API

target/native/x86_64-linux/
  bin/
    liboptixjni.so          # Compiled JNI shared library
    sphere_combined.ptx     # Compiled CUDA kernels
```

**Build process:**
1. sbt-jni plugin detects `src/main/native/CMakeLists.txt`
2. CMake compiles C++ code and CUDA shaders to PTX
3. Google Test suite runs (16 C++ unit tests)
4. Shared library (`liboptixjni.so`) and PTX files copied to `target/native/*/bin/`
5. Scala code loads native library via functional-style loader with Try monad

**Key build files:**
- `build.sbt`: Configures sbt-jni plugin, CMake flags for quiet builds, PTX file copying
- `project/CMakeWithoutVersionBug.scala`: Custom CMake build tool (fixes sbt-jni version parsing bug)
- `optix-jni/src/main/native/CMakeLists.txt`: CMake configuration with quiet install messages

**Note on build verbosity**: sbt-jni runs CMake on every compile (no sbt-level incremental tracking), but CMake itself handles dependency tracking and skips actual compilation when source files haven't changed. Build output is minimal thanks to `-Wno-dev`, `--log-level=WARNING`, and `CMAKE_INSTALL_MESSAGE LAZY`.

### 4D Rendering Pipeline

4D objects go through a two-stage transformation:
1. **4D Rotation**: Applied in 4D space (XW, YW, ZW plane rotations)
2. **4D→3D Projection**: Projects 4D vertices onto 3D "screen" using perspective projection with
   configurable screen and eye distances

The `RotatedProjection` and `FractionalRotatedProjection` classes wrap 4D objects (`Mesh4D`) and
handle these transformations before generating 3D `ModelInstance` objects for LibGDX.

### Fractional Levels

Fractional levels (e.g., `--level 1.5`) are implemented using alpha blending:
- Renders both floor level (e.g., 1) and ceiling level (e.g., 2)
- Floor level alpha transitions from 1.0 (opaque) to 0.0 (transparent) as fractional part
  approaches 1.0
- Ceiling level rendered fully opaque
- Implemented via `FractionalLevelSponge` trait and `FractionalRotatedProjection` wrapper

### Input Handling

```
menger.input/
  - EventDispatcher           # Publishes rotation/projection events to observers
  - Observer                  # Trait for event subscribers (implemented by Geometry)
  - CameraController          # Handles camera movement
  - KeyController             # Handles 4D rotation and projection keyboard controls
  - MengerInputMultiplexer    # Combines multiple input processors
```

Uses Observer pattern: `Geometry` objects subscribe to parameter changes from `EventDispatcher`,
which is triggered by `KeyController` during interactive mode.

## OptiX Transparency and Refraction Physics

### Fresnel Reflectance

Surface reflectance is calculated using the **Schlick approximation** of the Fresnel equations:

```
R₀ = ((n₁ - n₂) / (n₁ + n₂))²
R(θ) = R₀ + (1 - R₀)(1 - cos θ)⁵
```

Where:
- `n₁` = IOR of surrounding medium (assumed to be 1.0 for air/vacuum)
- `n₂` = IOR of sphere material (from `--ior` parameter)
- `θ` = angle between ray direction and surface normal
- `R₀` = reflectance at perpendicular incidence (0° angle)
- `R(θ)` = reflectance at angle θ

**Example values:**
- Glass (IOR = 1.5): R₀ = 0.04 (4% reflection perpendicular, up to 100% at grazing angles)
- Water (IOR = 1.33): R₀ = 0.02 (2% reflection perpendicular)
- Diamond (IOR = 2.42): R₀ = 0.17 (17% reflection perpendicular)

### Beer-Lambert Law (Volume Absorption)

Light intensity decreases exponentially as it travels through an absorbing medium:

```
I(d) = I₀ · exp(-α · d)
```

Where:
- `I₀` = initial light intensity
- `d` = distance traveled through medium
- `α` = absorption coefficient (derived from color alpha and RGB values)

**Color interpretation:**
- **Alpha value**: Controls opacity/absorption strength (0.0 = fully transparent/no absorption, 1.0 = fully opaque/maximum absorption)
- **RGB values**: Control wavelength-dependent absorption (color tint)
  - Example: `color=#00ff8080` (semi-transparent green-cyan)
    - RGB (0, 255, 128) creates green-cyan tint
    - Alpha 0.5 means 50% opacity (semi-transparent, moderate absorption)
  - Pure white `#ffffffff` = no tint, fully opaque
  - Pure white `#ffffff00` = no tint, fully transparent (no absorption)

**Implementation:**
- For each wavelength (R, G, B), calculate absorption coefficient:
  ```
  α_r = -log(color.r) * color.a
  α_g = -log(color.g) * color.a
  α_b = -log(color.b) * color.a
  ```
  Where `color.a` is the alpha/opacity value (0.0 = transparent, 1.0 = opaque)
- Apply Beer-Lambert law during ray traversal through sphere volume

### Snell's Law (Refraction)

When a ray enters/exits the sphere, it refracts according to Snell's law:

```
n₁ · sin(θ₁) = n₂ · sin(θ₂)
```

Where:
- `θ₁` = angle of incidence
- `θ₂` = angle of refraction
- `n₁`, `n₂` = IORs of the two media

**Refraction direction:**
```
r = η · d + (η · cos(θ₁) - cos(θ₂)) · n
```

Where:
- `r` = refracted ray direction
- `d` = incident ray direction
- `n` = surface normal
- `η = n₁ / n₂` = relative IOR

## Future Enhancements

### Phase 3: Refraction and Fresnel Reflection (In Progress)

Add physically-based refraction and reflection to OptiX sphere rendering.

**CLI Parameters:**
- `--ior <value>`: Index of refraction (must be > 0, default 1.0 for no refraction)
  - Common values: glass=1.5, water=1.33, diamond=2.42, air=1.0

**Implementation phases:**
1. Add `--ior` CLI parameter with validation (> 0)
2. Pass IOR through Scala layers to OptiX renderer
3. Update CUDA shaders to:
   - Calculate Fresnel reflectance using Schlick approximation
   - Implement Snell's law for refraction at entry/exit points
   - Apply Beer-Lambert law for volume absorption
4. Add tests for various IOR values and color combinations

## Code Quality Rules

### Wartremover (Compile-Time Checks)

Enabled errors:
- `Wart.Var` - No mutable variables (use `@SuppressWarnings` if required for LibGDX integration)
- `Wart.While` - No while loops (use functional alternatives)
- `Wart.AsInstanceOf`, `Wart.IsInstanceOf` - No unsafe casting/type checks
- `Wart.Throw` - No exception throwing (use Try/Either)

Note: `Wart.Null` and `Wart.Return` are disabled for LibGDX compatibility.

### Scalafix Rules (.scalafix.conf)

- `OrganizeImports` - Sort imports: java/javax → scala → other
- `DisableSyntax` - Enforces `noNulls = true`, `noReturns = true`, `noXml = true`
- `LeakingImplicitClassVal`, `NoValInForComprehension` - Prevent common pitfalls
- `RedundantSyntax` - Scala 3 migration helpers

### Compiler Flags

- `-Wunused:imports` - Fail on unused imports
- `-deprecation`, `-feature` - Warn on deprecated/experimental features
- `-explain` - Provide detailed error explanations

## Development Checklist

**After making changes:**
1. Add changes to CHANGELOG.md under "Added", "Fixed", or "Upgraded" headers
2. Follow keepachangelog.com format
3. Never use `git add -A` or `git commit -a` - always add files explicitly
4. Keep line lengths ≤100 characters (unless it significantly reduces readability)
5. Run `sbt compile && sbt test --warn` before committing (includes OptiX JNI compilation and tests)
6. Run `sbt "scalafix --check"` to verify code quality

**For OptiX JNI changes:**
- Changes to C++/CUDA code in `optix-jni/src/main/native/` require CMake and CUDA toolkit
- Test with `sbt "project optixJni" test` to verify native compilation and JNI bindings
- Clean native build if needed: `rm -rf optix-jni/target/native`
- The native library is loaded dynamically at runtime, so failures may not appear until tests run

## Troubleshooting

### OptiX JNI Build Issues

**CMake warnings about "Ignoring extra path":**
- This warning has been fixed by using a custom `CMakeWithoutVersionBug` build tool
- See `project/CMakeWithoutVersionBug.scala` for implementation details
- The issue was caused by sbt-jni passing the cmake version number (e.g., "328") as an extra argument

**"library not found" or "cannot find -lcuda" errors:**
- Ensure CUDA toolkit is installed: `nvcc --version`
- Check `LD_LIBRARY_PATH` includes CUDA libs: `echo $LD_LIBRARY_PATH`
- On Ubuntu/Debian: `sudo apt-get install nvidia-cuda-toolkit`

**"OptiX headers not found" during compilation:**
- Set `OPTIX_ROOT` environment variable to OptiX SDK path
- CMakeLists.txt auto-detects SDK and prefers highest version (9.0 over 8.0)
- Download from: https://developer.nvidia.com/optix

**CUDA error 718 ("invalid program counter") during GPU execution:**
- **Cause**: OptiX SDK version mismatch with driver's OptiX runtime version
- **Symptom**: Tests fail with `cudaDeviceSynchronize() failed: invalid program counter (718)`
- **Diagnosis**: Check driver's OptiX version vs SDK version:
  ```bash
  # Check driver's OptiX version
  strings /usr/lib/x86_64-linux-gnu/libnvoptix.so.* | grep "OptiX Version"

  # Check which SDK version was used to build
  grep "OptiX SDK:" optix-jni/target/native/x86_64-linux/build/CMakeCache.txt
  ```
- **Fix**: Install matching OptiX SDK version and rebuild:
  ```bash
  # For driver 580.x+ (OptiX 9.0), download SDK 9.0 from NVIDIA
  rm -rf optix-jni/target/native
  sbt "project optixJni" compile
  ```
- **Prevention**: CMakeLists.txt now auto-detects and prefers highest SDK version
- **Root cause**: OptiX has strict ABI compatibility - SDK must match driver runtime

**"UnsatisfiedLinkError: no optixjni in java.library.path":**
- The native library didn't build or isn't in the expected location
- Check `optix-jni/target/native/x86_64-linux/bin/liboptixjni.so` exists
- Run `sbt "project optixJni" nativeCompile` to rebuild
- For tests, `build.sbt` sets the library path automatically via `Test / javaOptions`

**CUDA Architecture Support:**
- The native library (`liboptixjni.so`) contains C++ host code (no device code)
  - Calls CUDA/OptiX APIs but doesn't contain GPU kernels
  - Works on any system with compatible CUDA runtime
- OptiX shaders (`sphere_combined.ptx`) are PTX intermediate representation:
  - Targets compute_52 (Maxwell generation) as minimum
  - PTX is JIT-compiled at runtime to actual GPU architecture
  - Single PTX file works on any NVIDIA GPU from 2014 onwards (sm_52, 75, 86, 89, etc.)
  - OptiX requirement: Can only target one architecture; we use virtual for maximum compatibility

**"CMake Error: The source ... does not match ... used to generate cache" after Docker builds:**
- **Cause**: CMake cache was created in Docker container at different path (e.g., `/builds/lilacashes/menger`) but you're now building locally
- **Automatic fix**: As of this commit, build.sbt automatically detects and cleans mismatched CMake caches
- **Manual fix**: `pkexec chown -R $USER:$USER optix-jni/target/ && rm -rf optix-jni/target/native`
- **Prevention**: Always run Docker containers with your user ID (see "Testing Docker Images Locally" section below)

**"AccessDeniedException" when writing class files after Docker builds:**
- **Cause**: Docker container ran as root and created root-owned files in `optix-jni/target/`
- **Fix**: `pkexec chown -R $USER:$USER optix-jni/target/`
- **Prevention**: Always run Docker containers with `--user $(id -u):$(id -g)` (see below)

**"Failed to open PTX file" or solid red rendering after sbt clean:**
- **Cause**: PTX file compiled to `optix-jni/target/classes/native/x86_64-linux/sphere_combined.ptx` but OptiX runtime looks for it at `target/native/x86_64-linux/bin/sphere_combined.ptx`
- **Symptom**: Rendering shows solid red image instead of proper sphere, with errors `[OptiX] Render failed: Failed to open PTX file: target/native/x86_64-linux/bin/sphere_combined.ptx (errno: 2 - No such file or directory)`
- **Root Cause**: After `sbt clean`, the `target/` directory is removed. The build system compiles PTX to `optix-jni/target/classes/native/` but OptiX runtime expects it in `target/native/x86_64-linux/bin/`
- **Fix**: Copy PTX file to runtime location:
  ```bash
  mkdir -p target/native/x86_64-linux/bin
  cp optix-jni/target/classes/native/x86_64-linux/sphere_combined.ptx target/native/x86_64-linux/bin/
  ```
- **When OptiX fails to load PTX**: It throws a runtime exception with "Failed to open PTX file" error. Check that the PTX file exists at the expected location

**Wrong shader file being edited:**
- **Issue**: The separate shader files (`sphere_miss.cu`, `sphere_closesthit.cu`, `sphere_raygen.cu`) in `optix-jni/src/main/native/shaders/` are NOT compiled or used
- **Correct file**: Edit `sphere_combined.cu` which contains all three shaders in one file
- **How to verify**: Check `optix-jni/src/main/native/CMakeLists.txt` - it specifies `shaders/sphere_combined.cu` as the compilation target
- **Why separate files exist**: They are outdated from an earlier implementation before the combined shader approach

## NVIDIA GPU Development

The project includes Terraform configuration for launching AWS EC2 GPU spot instances for
CUDA/OptiX development. See terraform/README.md for details.

Quick reference:
```bash
# Build AMI (one-time)
./scripts/build-ami.sh /path/to/NVIDIA-OptiX-SDK-installer.sh

# Launch instance
./scripts/nvidia-spot.sh --ami-id ami-xxxxxxxxxxxx

# List instance types
./scripts/nvidia-spot.sh --list-instances

# Show running instances
./scripts/nvidia-spot.sh --list-running

# Terminate instance
./scripts/nvidia-spot.sh --terminate <instance-id>
```

Features: Auto-termination on logout, git identity auto-configuration, X11 forwarding, cost
controls.

## CI/CD Configuration

### GitLab Runner Setup for OptiX JNI Tests

The `Test:OptiXJni` CI job requires a GitLab Runner with GPU support and OptiX library access. See `optix-jni/RUNNER_SETUP.md` for complete installation instructions.

**Quick setup summary:**

1. **Install NVIDIA Driver** (580.x or newer recommended)
2. **Install Docker Engine** (19.03+)
3. **Install NVIDIA Container Toolkit**:
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

4. **Install GitLab Runner**:
   ```bash
   curl -L "https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh" | sudo bash
   sudo apt-get install gitlab-runner
   ```

5. **Register Runner** with tag `nvidia`:
   ```bash
   sudo gitlab-runner register
   # URL: https://gitlab.com/
   # Token: (from project Settings > CI/CD > Runners)
   # Tags: nvidia
   # Executor: docker
   # Default image: ubuntu:24.04
   ```

6. **Configure Runner for GPU and OptiX** - Edit `/etc/gitlab-runner/config.toml`:
   ```toml
   [[runners]]
     name = "nvidia-gpu-runner"
     url = "https://gitlab.com/"
     token = "YOUR_RUNNER_TOKEN"
     executor = "docker"
     tags = ["nvidia"]
     [runners.docker]
       tls_verify = false
       image = "ubuntu:24.04"
       privileged = false
       volumes = ["/cache", "/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro"]
       gpus = "all"
   ```

   **Critical configuration**:
   - `gpus = "all"` - Enables GPU access in containers
   - Volume mount for `libnvoptix.so.1` - Required for OptiX runtime (see troubleshooting below)

7. **Restart Runner**:
   ```bash
   sudo gitlab-runner restart
   ```

### OptiX Library Requirement

**Important**: OptiX 7.0+ does not ship the runtime library with the SDK. The OptiX runtime (`libnvoptix.so`) comes from the NVIDIA driver on the host and must be mounted into containers.

The NVIDIA Container Toolkit does not automatically mount OptiX libraries (the `optix` capability is not universally supported). Therefore, the OptiX library must be explicitly mounted as a volume in the runner configuration.

**Verify OptiX library exists on host**:
```bash
ls -la /usr/lib/x86_64-linux-gnu/libnvoptix.so.1
```

If missing, update your NVIDIA driver to a version with OptiX support (580.x+ recommended).

### Docker Image Build and Push

When updating the OptiX CI Docker image:

```bash
# Set version tag (format: {CUDA}-{OptiX}-{Java}-{sbt})
export VERSION=12.8-9.0-25-1.11.7

# Copy OptiX SDK installer to build context
cp ~/Downloads/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh optix-jni/

# Build the image
docker build -t registry.gitlab.com/lilacashes/menger/optix-cuda:$VERSION -f optix-jni/Dockerfile optix-jni/

# Tag as 'latest'
docker tag registry.gitlab.com/lilacashes/menger/optix-cuda:$VERSION registry.gitlab.com/lilacashes/menger/optix-cuda:latest

# Login and push
docker login registry.gitlab.com
docker push registry.gitlab.com/lilacashes/menger/optix-cuda:$VERSION
docker push registry.gitlab.com/lilacashes/menger/optix-cuda:latest

# Update version in .gitlab-ci.yml
# Set OPTIX_DOCKER_VERSION: 12.8-9.0-25-1.11.7
```

**Image architecture:**
- Base: `nvidia/cuda:12.8.0-devel-ubuntu24.04` (~9GB from DockerHub)
- Layer 2: Build tools (cmake, g++, ~200MB)
- Layer 3: OptiX SDK 9.0 (~500MB)
- Layer 4: Java 25 LTS (~400MB)
- Layer 5: sbt 1.11.7 (~100MB)

### Testing Docker Images Locally

**IMPORTANT**: When testing CI Docker images locally, ALWAYS run containers with your user ID to prevent permission issues. Docker containers run as root by default, which creates root-owned files in mounted directories that cause "AccessDeniedException" and CMake cache conflicts.

**Correct way to test OptiX CI image locally:**

```bash
# Test compilation (no GPU required for compilation)
docker run --rm \
  --user $(id -u):$(id -g) \
  -v "$PWD:/workspace" \
  -w /workspace \
  registry.gitlab.com/lilacashes/menger/optix-cuda:12.8-9.0-25-1.11.7 \
  bash -c "sbt 'project optixJni' compile"

# Test with GPU access (requires nvidia-container-toolkit)
docker run --rm \
  --user $(id -u):$(id -g) \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -v "$PWD:/workspace" \
  -w /workspace \
  registry.gitlab.com/lilacashes/menger/optix-cuda:12.8-9.0-25-1.11.7 \
  bash -c "sbt 'project optixJni' test"
```

**Key flags:**
- `--user $(id -u):$(id -g)` - **CRITICAL**: Run as your user, not root
- `-v "$PWD:/workspace"` - Mount current directory as /workspace (not /builds/lilacashes/menger)
- `-w /workspace` - Set working directory inside container
- `--gpus all` - Enable GPU access (for runtime tests only)
- `-e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility` - Mount OptiX/RTX libraries

**What NOT to do:**

```bash
# ❌ WRONG - runs as root, creates permission issues
docker run --rm -v "$PWD:/builds/lilacashes/menger" ...

# ❌ WRONG - mounts at GitLab CI path, causes CMake cache conflicts
docker run --rm --user $(id -u):$(id -g) -v "$PWD:/builds/lilacashes/menger" ...
```

**If you already have permission issues from a previous Docker run:**

```bash
# Fix ownership of root-owned files
pkexec chown -R $USER:$USER optix-jni/target/

# Clean CMake cache (or let build.sbt auto-clean on next compile)
rm -rf optix-jni/target/native
```

### Troubleshooting CI Failures

#### Test:OptiXJni job fails with "OPTIX_ERROR_LIBRARY_NOT_FOUND" or "Error initializing RTX library"

**Cause**: OptiX 7.0+ runtime libraries (`libnvoptix.so`, `libnvidia-rtcore.so`) not accessible in container. These libraries are part of the NVIDIA driver, not the OptiX SDK.

**Solution**: The CI job needs access to NVIDIA driver libraries through proper container configuration:

1. **In .gitlab-ci.yml**, ensure the job has:
   ```yaml
   Test:OptiXJni:
     variables:
       NVIDIA_DRIVER_CAPABILITIES: "graphics,compute,utility"  # Critical!
     before_script:
       # Create symlink for RTX library if needed
       - |
         if [ -f /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.* ] && [ ! -f /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.1 ]; then
           ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.* /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.1
         fi
       - ldconfig || true
   ```

2. **In GitLab Runner config** (`/etc/gitlab-runner/config.toml`), ensure:
   ```toml
   [[runners]]
     [runners.docker]
       gpus = "all"  # Enable GPU access
   ```

3. **Verify libraries on host**:
   ```bash
   ls -la /usr/lib/x86_64-linux-gnu/libnvoptix.so*
   ls -la /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so*
   ```

**Background**: The `NVIDIA_DRIVER_CAPABILITIES` environment variable tells the NVIDIA Container Toolkit which driver libraries to mount into the container. Setting it to `"graphics,compute,utility"` automatically mounts the required OptiX and RTX libraries.

#### Job fails with "could not select device driver with capabilities: [[gpu]]"

**Cause**: NVIDIA Container Toolkit not configured.

**Solution**:
1. Install NVIDIA Container Toolkit (see setup instructions above)
2. Reconfigure Docker: `sudo nvidia-ctk runtime configure --runtime=docker`
3. Restart Docker: `sudo systemctl restart docker`
4. Restart runner: `sudo gitlab-runner restart`

#### Runner not picking up jobs with 'nvidia' tag

**Cause**: Runner not tagged correctly.

**Solution**:
1. Check runner tags in GitLab UI: Project Settings > CI/CD > Runners
2. Add 'nvidia' tag to runner if missing
3. Or edit `/etc/gitlab-runner/config.toml` and add `tags = ["nvidia"]`
4. Restart runner

#### Job runs on wrong runner

**Cause**: Multiple runners with same tag, wrong one has priority.

**Solution**:
1. Pause unwanted runners via GitLab UI or API:
   ```bash
   glab api --method PUT projects/lilacashes%2Fmenger/runners/<RUNNER_ID> -f "paused=true"
   ```
2. Or configure runner priority in GitLab UI (if available)

#### Docker fails after adding OptiX to supported-driver-capabilities

**Issue**: Adding `"optix"` to `supported-driver-capabilities` in `/etc/nvidia-container-runtime/config.toml` causes Docker containers to fail with "unsupported capabilities" error.

**Root cause**: The `optix` capability is not recognized by all versions of NVIDIA Container Toolkit.

**Solution**: Do NOT add `optix` to supported-driver-capabilities. Use `NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility` environment variable instead (see solution above).

#### CMake cache conflicts when testing locally

**Issue**: When building locally then testing in Docker, you may see CMake errors about incorrect source directories.

**Root cause**: CMake caches absolute paths. When the build directory is mounted in Docker at a different path, the cache becomes invalid.

**Solution**: Clean the native build directory before running Docker tests:
```bash
rm -rf optix-jni/target/native
```

Note: This is only an issue for local development. GitLab CI automatically cleans the workspace.

### CUDA Architecture Compatibility

The OptiX JNI build uses **CUDA architecture sm_75** (set in `CMakeLists.txt` line 38), which targets RTX 20 series GPUs. This is **forward-compatible** with newer GPUs:

| GPU | Compute Capability | Compatible with sm_75 PTX |
|-----|-------------------|---------------------------|
| RTX 2080 Ti | sm_75 | ✓ Native |
| RTX 3090 | sm_86 | ✓ JIT compilation |
| RTX 4090 | sm_89 | ✓ JIT compilation |
| RTX A1000 | sm_86 | ✓ JIT compilation |
| Tesla T4 | sm_75 | ✓ Native |

PTX (Parallel Thread Execution) is NVIDIA's virtual assembly language that provides forward compatibility. When PTX compiled for sm_75 runs on sm_86+ GPUs, the driver JIT-compiles it to the native architecture, ensuring compatibility with future GPUs.
- ALWAYS run the test suite before committing
- always run the pre-push hook before committing to detect errors early and not commit broken code
- after pushing to the remote git repository, always monitor the pipeline for failures
- do not add docstrings unless they really add information that is not clear from the names and types of the method
- keep lines to 100 charactes max
- one import per line as enforced by scalafix
- aim for functional programming style. in particular avoid mutable variables (var) and throwing exceptions unless it is absolutely necessary and dictated by the interface.
- when writing tests, use AnyFlatSpec as the testing style
- This is a Scala 3 project. Always prefer Scala 3 style over Scala 2 style.