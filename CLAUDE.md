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

### Basic Development

```bash
# Compile (includes native OptiX JNI compilation)
sbt compile

# Run tests (both Scala and OptiX JNI tests)
sbt test --warn

# Test only the core Scala application
sbt "project root" test --warn

# Test only OptiX JNI bindings (requires CUDA/OptiX)
sbt "project optixJni" test --warn

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

The OptiX JNI bindings are in the `optix-jni` subproject and require:
- CUDA Toolkit 12.0+
- NVIDIA OptiX SDK 8.0
- CMake 3.18+
- C++ compiler with C++17 support

```bash
# Work in the OptiX JNI subproject
sbt "project optixJni"

# Compile native code only
sbt "project optixJni" nativeCompile

# Clean and rebuild native code
rm -rf optix-jni/target/native && sbt "project optixJni" compile

# Run OptiX JNI tests
sbt "project optixJni" test

# View generated PTX files (CUDA kernels)
ls optix-jni/target/native/x86_64-linux/bin/*.ptx
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
- **Tests**: All 762+ tests pass (Scala: 750+, OptiX JNI: 12+)
- **Native artifacts**: Shared library (`liboptixjni.so`) and 3 PTX files generated
- **Packaging**: Creates ~22MB ZIP file
- **Integration**: All sponge types render successfully with expected face counts
- **Animation**: Generates 5 PNG files (800x600 RGBA)

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
