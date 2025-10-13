# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Menger is a Scala 3 application that renders fractals in 3D and 4D using LibGDX for graphics. It
generates Menger sponges (3D), tesseract sponges (4D), and their variants through recursive
subdivision algorithms. The project supports interactive exploration, animations, and fractional
iteration levels with alpha blending.

## Development Commands

### Basic Development

```bash
# Compile
sbt compile

# Run tests
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

- **Compilation**: Clean compile with no errors
- **Tests**: All 542+ tests pass
- **Packaging**: Creates ~22MB ZIP file
- **Integration**: All sponge types render successfully with expected face counts
- **Animation**: Generates 5 PNG files (800x600 RGBA)

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
5. Run `sbt compile && sbt test --warn` before committing
6. Run `sbt "scalafix --check"` to verify code quality

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
