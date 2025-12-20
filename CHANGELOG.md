# Changelog

## [0.4.0] - 2025-12-01

### Added
- **Instance Acceleration Structure (IAS)** - Multi-object rendering foundation
  - `addSphereInstance()` API for adding sphere instances with position and material
  - `addTriangleMeshInstance()` API for triangle mesh instances with transforms
  - Per-instance 4x3 transform matrices and material properties
  - GAS registry for geometry type management
  - `optixGetInstanceId()` for per-instance material lookup in shaders
- **Multi-Object CLI** - `--objects` parameter with keyword=value format
  - Support for sphere, cube, sponge-volume, sponge-surface, and cube-sponge types
  - Per-object position, size, color, IOR, and level parameters
  - Example: `--objects type=sphere:pos=-1,0,0:color=#FF0000`
- **Cube-Based Sponge (GPU Instancing)** - Memory-efficient sponge rendering
  - `CubeSpongeGenerator` generates instance transforms instead of merged geometry
  - One base cube mesh shared by all instances (up to 3.2M instances at level 5)
  - 40-80x memory reduction vs. merged mesh approach
  - CLI: `--objects type=cube-sponge:pos=0,0,0:level=2:color=#00FF00`
  - Configurable instance limit via `--max-instances` (default: 64)
- **Rendering Tests for IAS** - 18 tests including repeated render stress tests
- **Integration Tests** - Multi-object and triangle mesh rendering validation
- **Shadow Rays for Triangle Meshes** - Triangle meshes now cast shadows correctly
  - Shadow rays trace against IAS handle in multi-object mode
  - Shadow rays trace against GAS handle in single-object mode
  - Un-ignored test "cast shadows on the plane" now passes

### Fixed
- **CUDA error 700 in IAS mode** - Fixed use-after-free bug in GAS buffer management
  - IAS GAS buffers now managed separately from BufferManager
  - Multiple renders with IAS now work correctly
- **Triangle mesh shadows** - Triangle meshes now cast shadows on plane and other objects
  - params.handle correctly set to GAS/IAS handle depending on mode
  - Shadow ray shader works with both sphere and triangle geometry

## [0.3.9] - 2025-12-01

### Added
- **Triangle Mesh Support** - OptiX can now render triangle meshes (foundation for future geometry)
  - `setTriangleMesh()` API for uploading vertex/index buffers to GPU
  - Per-vertex normals for correct shading
  - Separate hit group programs for spheres and triangles
- **Cube Primitive** - First triangle mesh object, rendered via OptiX
  - 12 triangles with outward-facing normals
  - Demonstrates triangle intersection and shading

### Changed
- OptiX pipeline now supports both sphere and triangle geometry types
- SBT (Shader Binding Table) extended for multiple geometry hit groups

## [0.3.8] - 2025-11-28

### Added
- **Caustics Rendering** (experimental, deferred) - Progressive Photon Mapping groundwork
  - Architecture in place but algorithm issues identified
  - Enable with `--caustics` (may not produce visible results yet)

### Changed
- Refactored C++ architecture: decomposed OptiXWrapper into BufferManager and CausticsRenderer
- Enabled parallel test execution and enforced code quality tools across all subprojects

## [0.3.7] - 2025-11-21

### Added
- **Unified Color Type** - New `menger.common.Color` class for consistent color handling
  - RGBA components (0.0-1.0 range) with validation
  - `Color.fromRGB()`, `Color.fromRGBA()`, `Color.fromHex()` factory methods
  - `toRGBArray` and `toRGBAArray` methods for JNI conversion
- **Custom Plane Colors** - Configure plane colors via `--plane-color` flag
  - Solid color: `--plane-color #RRGGBB`
  - Checkered pattern: `--plane-color RRGGBB:RRGGBB`
- **Color Conversion Utilities** - Extension methods for LibGDX/common Color interop
  - `toCommonColor` extension on LibGDX Color
  - `toGdxColor` extension on menger.common.Color

### Changed
- Light colors now use `Color` type instead of `Vector[3]` for consistency
- `setSphereColor` float overloads are now private; use `Color` API instead

## [0.3.6] - 2025-11-20

### Added
- **Multiple Light Sources** - Configure up to 8 lights via `--light` flag
  - Format: `--light <type>:x,y,z[:intensity[:color]]`
  - Supports directional and point lights
  - Example: `--light directional:1,1,-1:2.0:ffffff --light point:0,5,0:3.0:ff0000`
- **Shadow Rendering** - Realistic hard shadows with `--shadows` flag
  - Transparent objects cast lighter shadows based on material alpha
  - Glass casts light shadows, opaque objects cast dark shadows

## [0.3.5] - 2025-11-17

### Added
- **Fresnel Reflection** - Realistic glass rendering with reflection and refraction blending

### Fixed
- Glass rendering now works correctly (previously showed only refraction)
- Improved transparency rendering accuracy for semi-transparent materials

## [0.3.4] - 2025-11-02

### Added
- **OptiX GPU Ray Tracing** - Hardware-accelerated sphere rendering with `--optix` flag
  - Configure sphere radius with `--sphere-radius <value>`
  - Save screenshots with `--save-name <filename>`
  - Auto-exit after timeout with `--timeout <seconds>`

## [0.3.3] - 2025-10-26

### Added
- OptiX ray tracing support for realistic lighting and materials

## [0.3.2] - 2025-10-23

### Added
- GPU rendering now optional (build works without CUDA/OptiX)

## [0.3.1] - 2025-10-21

### Added
- `--log-level` option to control logging verbosity (ERROR, WARN, INFO, DEBUG)
- `--fps-log-interval` option to control frequency of FPS logging

## [0.3.0] - 2025-10-06

### Added
- Remote GPU development support with AWS spot instances


## [0.2.9] - 2025-10-05

### Added
- Level animation support for all fractal sponge types via `--animate frames=N:level=start-end`
- Overlay rendering mode with `--face-color` and `--line-color` options for wireframe on transparent
  faces
- BlendingAttribute support in material builder for proper alpha transparency
- Comprehensive unit tests for level animation with single and chained animation specifications
- Validation to prevent parameters from being specified both as CLI option and in animation spec
- Validation to prevent conflicting color option combinations
- Documentation for animation parameters including level, rotation, 4D rotation, and projection
  settings
- Documentation for overlay mode with face and line color options
- Strict code quality enforcement with wartremover errors for Var, While, AsInstanceOf, IsInstanceOf,
  Throw
- Strict null checking with scalafix (noNulls = true)
- Compiler flag -Wunused:imports for continuous import validation
- @SuppressWarnings annotations for necessary vars in LibGDX integration and performance-critical code

### Fixed
- Refactored AnimationSpecification to eliminate code duplication in interpolation logic
- Fixed PushToGithub CI job by adding branch fetch before checkout
- Replaced mutable var with functional alternatives (AtomicBoolean, AtomicReference) where possible
- Replaced null checks with Option wrapper for LibGDX compatibility
- Removed 7 unused imports across the codebase
- Fixed scaladoc warnings by filtering out plugin options

### Upgraded
- sbt-scalafix 0.11.1 → 0.14.3
- logback-classic 1.5.18 → 1.5.19 (fixes CVE-2025-11226 security vulnerability)
- scalamock 7.4.1 → 7.5.0

## [0.2.8] - 2025-10-04

### Added
- Scalafix integration for code quality and automated refactoring
- Fractional level support for SpongeBySurface with smooth alpha transitions
- FractionalLevelSponge trait to eliminate code duplication between sponge implementations
- Fractional level support for TesseractSponge and TesseractSponge2 via FractionalRotatedProjection
  wrapper

### Fixed
- Path traversal vulnerability in screenshot filename handling with comprehensive test coverage
- Improved timing precision by replacing System.currentTimeMillis with System.nanoTime
- Made getIntegerModel a lazy val to prevent repeated sponge instantiation in render loop
- Corrected alpha calculation for fractional level sponges to properly transition from full opacity 
  to transparency
- Eliminated code duplication by moving createMaterialWithAlpha to FractionalLevelSponge companion 
  object

### Upgraded
- Updated dependencies: scala-logging 3.9.6, sbt-native-packager 1.11.3, sbt-scoverage 2.3.1, 
  sbt-jupiter-interface 0.11.3

## [0.2.7] - 2025-09-15

### Added
- `--color` option to set the color of the rendered object

### Upgraded
- Scala to 3.7.3
- sbt to 1.11.5

## [0.2.6] - 2025-08-12

### Added
- replaced LibGDX's `Vector4` and `Matrix4` with `Vector[4]` and `Matrix[4]` for future 
  extensibility

### Upgraded
- sbt to 1.11.4

## [0.2.5] - 2025-03-27

### Added
- script parameter animations
- Use named tuples throughout the code

### Upgraded
- Scala to 3.7.2
- sbt to 1.10.11
- Scalamock to 7.4.0

## [0.2.4] - 2025-03-20

### Added
- Clean up code by replacing Tuples with explicit classes

### Upgraded
- ScalaMock to 7.2.0, fixing resulting errors in tests

## [0.2.3] - 2025-03-17

### Added
- Visualize a four-dimensional Menger Sponge analog generated by subdividing each face into 16
  subfaces

### Upgraded
- Scala to 3.6.4
- Scalatest to 3.2.19

## [0.2.2] - 2024-10-30

### Upgraded
- Scala to 3.5.2
- Scalatest to 3.2.18
- Scallop to 5.1.0

## [0.2.1] - 2024-03-26

### Added
- Visualize a four-dimensional Menger Sponge analog generated by subdividing a Tesseract into 48 
  smaller Tesseracts
- Changelog with retroactive entries for previous versions

## [0.2.0] - 2024-03-19

### Added
- Visualize a Tesseract 
- Interactively rotate and change projection distance of the Tesseract

## [0.1.0] - 2024-02-20

### Initial Release
- Visualize Menger Sponge generated by subdividing a cube into 20 smaller cubes
- Visualize Menger Sponge generated by subdividing a face into 12 smaller faces


[0.4.0]: https://gitlab.com/lilacashes/menger/-/compare/0.3.9...0.4.0
[0.3.9]: https://gitlab.com/lilacashes/menger/-/compare/0.3.8...0.3.9
[0.3.8]: https://gitlab.com/lilacashes/menger/-/compare/0.3.7...0.3.8
[0.3.7]: https://gitlab.com/lilacashes/menger/-/compare/0.3.6...0.3.7
[0.3.6]: https://gitlab.com/lilacashes/menger/-/compare/0.3.5...0.3.6
[0.3.5]: https://gitlab.com/lilacashes/menger/-/compare/0.3.4...0.3.5
[0.3.4]: https://gitlab.com/lilacashes/menger/-/compare/0.3.3...0.3.4
[0.3.3]: https://gitlab.com/lilacashes/menger/-/compare/0.3.2...0.3.3
[0.3.2]: https://gitlab.com/lilacashes/menger/-/compare/0.3.1...0.3.2
[0.3.1]: https://gitlab.com/lilacashes/menger/-/compare/0.3.0...0.3.1
[0.3.0]: https://gitlab.com/lilacashes/menger/-/compare/0.2.9...0.3.0
[0.2.9]: https://gitlab.com/lilacashes/menger/-/compare/0.2.8...0.2.9
[0.2.8]: https://gitlab.com/lilacashes/menger/-/compare/0.2.7...0.2.8
[0.2.7]: https://gitlab.com/lilacashes/menger/-/compare/0.2.6...0.2.7
[0.2.6]: https://gitlab.com/lilacashes/menger/-/compare/0.2.5...0.2.6
[0.2.5]: https://gitlab.com/lilacashes/menger/-/compare/0.2.4...0.2.5
[0.2.4]: https://gitlab.com/lilacashes/menger/-/compare/0.2.3...0.2.4
[0.2.3]: https://gitlab.com/lilacashes/menger/-/compare/0.2.2...0.2.3
[0.2.2]: https://gitlab.com/lilacashes/menger/-/compare/0.2.1...0.2.2
[0.2.1]: https://gitlab.com/lilacashes/menger/-/compare/0.2.0...0.2.1
[0.2.0]: https://gitlab.com/lilacashes/menger/-/compare/0.1.0...0.2.0
[0.1.0]: https://gitlab.com/lilacashes/menger/-/commit/f90eee11
