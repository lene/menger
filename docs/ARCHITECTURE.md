# Architecture

## Core Components

**Main.scala**: Entry point. Creates LibGDX configuration, instantiates `InteractiveMengerEngine` (interactive) or `AnimatedMengerEngine` (animation) based on CLI options.

**MengerEngine**: Abstract base class. Contains `generateObject()` factory method creating geometry instances. Supports overlay mode for transparent faces with wireframe overlays.

**Geometry trait** (`menger.objects.Geometry`): Base trait for all renderable objects. Observer pattern for rotation/projection parameter changes. Implements `getModel: List[ModelInstance]`.

## Fractal Generation Strategy

**Surface subdivision** (not volume) for efficiency:
- Volume: O(20^n) for 3D - subdivide into 27 cubes, remove 7
- Surface: O(12^n) for 3D - subdivide faces into 9 squares, remove center, add 4 hole faces

Surface subdivision generates only outer surface with no internal faces.

## Object Hierarchy

```
menger.objects/
  - Geometry (trait)          # Base trait
  - Square, Cube              # Basic 2D/3D primitives
  - SpongeBySurface           # 3D Menger (12 faces/face)
  - SpongeByVolume            # 3D Menger (20 cubes/cube)
  - FractionalLevelSponge     # Fractional level support with alpha
  - Composite                 # Multiple geometries with overlay

  higher_d/
    - Tesseract               # 4D hypercube
    - TesseractSponge         # 4D sponge (48 tesseracts/tesseract)
    - TesseractSponge2        # 4D sponge (16 faces/face)
    - RotatedProjection       # 4D rotation + projection to 3D
    - FractionalRotatedProjection  # Fractional levels for 4D
    - Mesh4D                  # 4D mesh data structure
```

## OptiX JNI Architecture

Two-layer architecture for GPU ray tracing.

### Low-Level Layer: OptiXContext

Pure OptiX API wrapper, stateless, explicit resource management, 1:1 mapping to OptiX.

**Files:** `include/OptiXContext.h`, `OptiXContext.cpp`

**Key methods:**
- `initialize()`/`destroy()` - Device context lifecycle
- `createModuleFromPTX()`/`destroyModule()` - Shader compilation
- `createRaygenProgramGroup()`, `createMissProgramGroup()`, `createHitgroupProgramGroup()`
- `createPipeline()`/`destroyPipeline()` - Pipeline assembly
- `buildCustomPrimitiveGAS()`/`destroyGAS()` - Geometry acceleration
- `createRaygenSBTRecord()`, `createMissSBTRecord()`, `createHitgroupSBTRecord()`
- `launch()` - OptiX kernel execution

**Testing:** 16 Google Test C++ unit tests

### High-Level Layer: OptiXWrapper

Scene state management, convenience methods, performance optimization (scene data in Params not SBT). Uses OptiXContext via composition.

**Files:** `include/OptiXWrapper.h`, `OptiXWrapper.cpp`, `include/OptiXData.h`

**Key methods:**
- `setSphere()`, `setSphereColor()`, `setIOR()`, `setScale()` - Scene config
- `setCamera()`, `setLight()`, `setPlane()` - Environment
- `render()` - High-level rendering (builds pipeline, launches OptiX, returns RGBA)

### JNI Interface

Binds Scala `OptiXRenderer` to C++ `OptiXWrapper`. Per-instance handles, error propagation, functional-style loading with Try monad.

**Files:** `JNIBindings.cpp`, `menger/optix/OptiXRenderer.scala`

### Shaders (CUDA)

Single combined shader: `shaders/sphere_combined.cu` → PTX. Ray generation, miss, closest hit, custom sphere intersection. Reads scene data from Params (not SBT) for performance.

### Build Process

1. sbt-jni detects `CMakeLists.txt`
2. CMake compiles C++/CUDA to PTX
3. Google Test suite runs (16 tests)
4. Artifacts → `target/native/*/bin/`
5. Scala loads via functional loader with Try

**Build files:**
- `build.sbt` - sbt-jni config, CMake flags, PTX copying
- `project/CMakeWithoutVersionBug.scala` - Custom CMake tool (fixes version parsing)
- `optix-jni/src/main/native/CMakeLists.txt` - CMake config

**Note:** sbt-jni runs CMake every compile, but CMake skips unchanged files. Minimal output via `-Wno-dev`, `--log-level=WARNING`, `CMAKE_INSTALL_MESSAGE LAZY`.

## 4D Rendering Pipeline

Two-stage transformation:
1. **4D Rotation** - XW, YW, ZW plane rotations
2. **4D→3D Projection** - Perspective projection with configurable screen/eye distances

`RotatedProjection`/`FractionalRotatedProjection` wrap 4D objects (`Mesh4D`), generate 3D `ModelInstance` for LibGDX.

## Fractional Levels

Fractional levels (e.g., 1.5) via alpha blending:
- Floor level (1) alpha: 1.0→0.0 as fractional part→1.0
- Ceiling level (2) opaque
- Implemented via `FractionalLevelSponge` trait, `FractionalRotatedProjection` wrapper

## Input Handling

```
menger.input/
  - EventDispatcher           # Publishes rotation/projection events
  - Observer                  # Event subscribers (implemented by Geometry)
  - CameraController          # Camera movement
  - KeyController             # 4D rotation/projection keyboard
  - MengerInputMultiplexer    # Combines input processors
```

Observer pattern: `Geometry` subscribes to `EventDispatcher` triggered by `KeyController`.
