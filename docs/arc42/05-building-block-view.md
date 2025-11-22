# 5. Building Block View

## 5.1 Level 1: System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Menger System                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │                  │  │                  │  │                  │   │
│  │   menger.core    │  │  menger.objects  │  │   optix-jni      │   │
│  │   (Application)  │  │   (Geometry)     │  │   (Ray Tracing)  │   │
│  │                  │  │                  │  │                  │   │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘   │
│           │                     │                     │              │
│           └─────────────────────┴─────────────────────┘              │
│                                 │                                    │
│                                 ▼                                    │
│                    ┌────────────────────────┐                        │
│                    │     menger.common      │                        │
│                    │   (Shared utilities)   │                        │
│                    └────────────────────────┘                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 5.2 Level 2: Component Details

### 5.2.1 menger.core (Application Layer)

**Purpose:** Entry point, CLI parsing, engine orchestration.

```
menger/
├── Main.scala              # Entry point, LibGDX config
├── MengerCLIOptions.scala  # CLI argument parsing (Scallop)
├── MengerEngine.scala      # Abstract base, geometry factory
├── InteractiveMengerEngine # Interactive mode
├── AnimatedMengerEngine    # Animation mode
└── OptiXResources.scala    # OptiX renderer lifecycle
```

### 5.2.2 menger.objects (Geometry Layer)

**Purpose:** Fractal geometry generation via surface subdivision.

```
menger.objects/
├── Geometry (trait)              # Base trait for renderable objects
├── Face, Direction               # Face representation for subdivision
├── Square, Cube                  # Basic 2D/3D primitives
├── SpongeBySurface               # 3D Menger (12 faces/face)
├── SpongeByVolume                # 3D Menger (20 cubes/cube)
├── FractionalLevelSponge         # Fractional level support
└── higher_d/
    ├── Tesseract                 # 4D hypercube
    ├── TesseractSponge           # 4D sponge (48 tesseracts/tesseract)
    ├── TesseractSponge2          # 4D sponge (16 faces/face)
    ├── Mesh4D                    # 4D mesh data structure
    └── RotatedProjection         # 4D→3D projection wrapper
```

### 5.2.3 optix-jni (Ray Tracing Layer)

**Purpose:** GPU-accelerated ray tracing via NVIDIA OptiX.

```
optix-jni/
├── src/main/scala/menger/optix/
│   ├── OptiXRenderer.scala       # Scala JNI interface
│   └── geometry/
│       ├── CubeGeometry.scala    # Cube mesh generation
│       └── SpongeGeometry.scala  # Sponge mesh export
│
└── src/main/native/
    ├── include/
    │   ├── OptiXContext.h        # Low-level OptiX wrapper
    │   ├── OptiXWrapper.h        # High-level scene API
    │   └── OptiXData.h           # Data structures, Params
    ├── OptiXContext.cpp          # OptiX API implementation
    ├── OptiXWrapper.cpp          # Scene state management
    ├── JNIBindings.cpp           # Scala↔C++ bridge
    └── shaders/
        └── sphere_combined.cu    # CUDA shaders (raygen, hit, miss)
```

## 5.3 OptiX JNI Architecture (Level 3)

### Two-Layer Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Scala Layer                                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  OptiXRenderer.scala                                          │   │
│  │  - Native method declarations                                 │   │
│  │  - Type-safe Scala API                                        │   │
│  │  - Color/Material wrappers                                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ JNI
┌────────────────────────────────▼────────────────────────────────────┐
│                         C++ Layer                                    │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  OptiXWrapper (High-Level)                                    │   │
│  │  - Scene state management                                     │   │
│  │  - Convenience methods (setSphere, setCamera, render)         │   │
│  │  - Performance optimization (data in Params)                  │   │
│  └─────────────────────────────┬────────────────────────────────┘   │
│                                │ uses                                │
│  ┌─────────────────────────────▼────────────────────────────────┐   │
│  │  OptiXContext (Low-Level)                                     │   │
│  │  - Pure OptiX API wrapper                                     │   │
│  │  - Stateless, explicit resource management                    │   │
│  │  - 1:1 mapping to OptiX functions                             │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Data Structures

| Structure | Location | Purpose |
|-----------|----------|---------|
| `Params` | OptiXData.h | Launch parameters (scene, camera, lights) |
| `HitGroupData` | OptiXData.h | Per-geometry SBT data |
| `Light` | OptiXData.h | Light source definition |
| `RayStats` | OptiXData.h | Ray tracing statistics |

## 5.4 Input Handling

```
menger.input/
├── EventDispatcher           # Publishes rotation/projection events
├── Observer                  # Event subscribers (implemented by Geometry)
├── CameraController          # Camera movement (LibGDX)
├── OptiXCameraController     # Camera control (OptiX window)
├── KeyController             # 4D rotation keyboard controls
└── MengerInputMultiplexer    # Combines input processors
```

**Pattern:** Observer pattern for 4D parameter changes. Geometry objects subscribe to EventDispatcher.
