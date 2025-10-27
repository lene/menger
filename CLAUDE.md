# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Menger is a Scala 3 application that renders fractals in 3D and 4D using LibGDX for graphics. It
generates Menger sponges (3D), tesseract sponges (4D), and their variants through recursive
subdivision algorithms. The project supports interactive exploration, animations, and fractional
iteration levels with alpha blending.

The project consists of two main components:
- **Core Scala application** (`menger`): LibGDX-based fractal renderer
- **OptiX JNI bindings** (`optix-jni`): Native CUDA/OptiX integration for GPU-accelerated ray
  tracing (optional, requires NVIDIA GPU)

## Development Commands

**Note**: Commands in this document use `sudo` for elevated privileges where needed. If your system doesn't have `sudo` configured (e.g., some Ubuntu configurations), use `pkexec` instead (e.g., `pkexec systemctl restart docker` instead of `sudo systemctl restart docker`).

### Basic Development

**Default build (without OptiX JNI):**

By default, the project builds **without** OptiX JNI support. This allows development on
systems without NVIDIA GPU/CUDA installed. The OptiX renderer will use a stub implementation
that returns placeholder data.

```bash
# Compile (core Scala only, NO OptiX JNI)
sbt compile

# Run tests (core Scala only, NO OptiX JNI)
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

### OptiX JNI Development (Optional)

The OptiX JNI bindings are **optional** and disabled by default. To enable them, set the
`ENABLE_OPTIX_JNI` environment variable to `true`. This requires:

- CUDA Toolkit 12.0+
- NVIDIA OptiX SDK 8.0
- CMake 3.18+
- C++ compiler with C++17 support

**Building with OptiX JNI enabled:**

```bash
# Enable OptiX JNI for a single command
ENABLE_OPTIX_JNI=true sbt compile

# Or export it for your entire session
export ENABLE_OPTIX_JNI=true
sbt compile

# Compile native code only
ENABLE_OPTIX_JNI=true sbt "project optixJni" nativeCompile

# Clean and rebuild native code
rm -rf optix-jni/target/native && ENABLE_OPTIX_JNI=true sbt "project optixJni" compile

# Run OptiX JNI tests
ENABLE_OPTIX_JNI=true sbt "project optixJni" test

# View generated PTX files (CUDA kernels) - only created when OptiX available
ls optix-jni/target/native/x86_64-linux/bin/*.ptx

# Package with OptiX JNI included
ENABLE_OPTIX_JNI=true sbt "Universal / packageBin"
```

**Note:** If CUDA/OptiX are not installed, the native library will build as a stub that
always returns placeholder data. The CMake build will show a summary of what was detected:

```
========================================
OptiX JNI Build Configuration:
  CUDA Support:   FALSE
  OptiX Support:  FALSE

  NOTE: Building stub library without GPU support
  The OptiXRenderer will use fallback implementation
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

## CMake Wrapper Setup (Required for OptiX JNI)

The project includes OptiX JNI bindings that require CMake for compilation. Due to a version
parsing bug in sbt-jni (the build plugin), CMake emits an annoying but harmless warning during
compilation. To suppress this warning, the project includes a `cmake` wrapper script that filters
it out.

**One-time setup after cloning the repository:**

```bash
mkdir -p ~/.local/bin
ln -sf "$PWD/cmake" ~/.local/bin/cmake
```

This creates a symlink in `~/.local/bin` (which is typically early in your PATH) that points to
the wrapper script. The wrapper calls the real `/usr/bin/cmake` but filters out the warning line.

**Notes:**
- This is automatically configured in CI/CD pipelines and AMI builds
- The wrapper is a simple bash script in the project root that filters stderr
- If you don't set this up, builds will still work but show a "CMake Warning: Ignoring extra path
  from command line" message
- The warning is caused by sbt-jni incorrectly parsing CMake version numbers (e.g., "3.28.3"
  becomes "328" appended to the build path)

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

The `optix-jni` subproject provides JNI bindings to NVIDIA OptiX for GPU-accelerated ray tracing:

```
optix-jni/src/main/
  native/                    # C++/CUDA source code
    CMakeLists.txt          # CMake build configuration
    OptiXWrapper.cpp        # C++ OptiX integration layer
    JNIBindings.cpp         # JNI interface to Scala
    include/
      OptiXWrapper.h        # C++ headers
    shaders/                # CUDA OptiX shaders (compiled to PTX)
      sphere_raygen.cu      # Ray generation shader
      sphere_closesthit.cu  # Closest hit shader
      sphere_miss.cu        # Miss shader

  scala/
    optix/                  # Scala JNI interface
      OptiXRenderer.scala   # Main Scala API
      OptiXNative.scala     # JNI native method declarations

target/native/x86_64-linux/
  bin/
    liboptixjni.so         # Compiled JNI shared library
    *.ptx                  # Compiled CUDA kernels
```

**Build process:**
1. sbt-jni plugin detects `src/main/native/CMakeLists.txt`
2. CMake compiles C++ code and CUDA shaders to PTX
3. Shared library (`liboptixjni.so`) and PTX files copied to `target/native/*/bin/`
4. Scala code loads native library via `System.loadLibrary()`

**Key files:**
- `build.sbt`: Configures sbt-jni plugin, sets library path for tests
- `cmake` (project root): Wrapper script to suppress sbt-jni version parsing warnings
- `optix-jni/src/main/native/CMakeLists.txt`: Defines CMake build for native code

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

## Future Enhancements

### Phase 3: Trait-Based Renderer Architecture (Planned)

The current OptiX JNI implementation uses a single `OptiXRenderer` class that directly calls
native methods. A future enhancement would refactor this into a trait-based architecture for
better separation of concerns and testability.

**Proposed design:**

```scala
// Common interface for all renderer implementations
trait Renderer {
  def initialize(): Boolean
  def setSphere(x: Float, y: Float, z: Float, radius: Float): Unit
  def setCamera(eye: Array[Float], lookAt: Array[Float], up: Array[Float], fov: Float): Unit
  def setLight(direction: Array[Float], intensity: Float): Unit
  def render(width: Int, height: Int): Array[Byte]
  def dispose(): Unit
  def isAvailable: Boolean
}

// Native OptiX implementation (when OptiX available)
class OptiXNativeRenderer extends Renderer {
  @native def initialize(): Boolean
  // ... other @native methods
}

// Pure Scala stub implementation (always available)
class OptiXStubRenderer extends Renderer {
  def initialize(): Boolean = { ... }
  def render(width: Int, height: Int): Array[Byte] = {
    // Returns gray placeholder or simple CPU ray tracer
  }
  // ... other stub methods
}

// Factory with automatic selection
object OptiXRenderer {
  def apply(): Renderer = {
    if (isLibraryLoaded) new OptiXNativeRenderer()
    else new OptiXStubRenderer()
  }
}
```

**Benefits:**
- Clean separation between native and stub implementations
- Stub compiles even if native library build fails
- Easy to add additional implementations (e.g., CPU ray tracer, Vulkan renderer)
- Better testability - can mock/test implementations independently
- Follows patterns from Netty (native epoll vs NIO) and LWJGL

**Estimated effort:** 2-3 hours

This architecture would be particularly useful when OptiX integration becomes more complex or
when adding alternative rendering backends.

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
- This is a known sbt-jni version parsing bug
- Set up the cmake wrapper: `ln -sf "$PWD/cmake" ~/.local/bin/cmake`
- See "CMake Wrapper Setup" section above

**"library not found" or "cannot find -lcuda" errors:**
- Ensure CUDA toolkit is installed: `nvcc --version`
- Check `LD_LIBRARY_PATH` includes CUDA libs: `echo $LD_LIBRARY_PATH`
- On Ubuntu/Debian: `sudo apt-get install nvidia-cuda-toolkit`

**"OptiX headers not found" during compilation:**
- Set `OPTIX_ROOT` environment variable to OptiX SDK path
- Default in CMakeLists.txt: `/usr/local/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64`
- Download from: https://developer.nvidia.com/optix

**"UnsatisfiedLinkError: no optixjni in java.library.path":**
- The native library didn't build or isn't in the expected location
- Check `optix-jni/target/native/x86_64-linux/bin/liboptixjni.so` exists
- Run `sbt "project optixJni" nativeCompile` to rebuild
- For tests, `build.sbt` sets the library path automatically via `Test / javaOptions`

**PTX compilation errors about multiple GPU architectures:**
- OptiX PTX shaders can only target one architecture at a time
- CMakeLists.txt sets `CMAKE_CUDA_ARCHITECTURES` to a single value (75 or 89)
- This is expected and correct for OptiX development

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