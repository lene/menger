# Changelog

## [Unreleased]

## [0.4.3] - 2026-02-05

### Added
- **Sprint Planning Reorganization** - Restructured sprints 10-14 based on completed work
  - Sprint 10: Scala DSL for Scene Description (prioritized for better workflow)
  - Sprint 11: 4D Framework Enhancements (remaining UX features)
  - Sprint 12: Visual Quality & Materials (new sprint from TODO priorities)
  - Sprint 13: Object Animation Foundation
  - Sprint 14: Advanced Animation System
  - Documentation: SPRINT_REORGANIZATION_2026-02.md explains rationale

### Changed
- **Test Performance** - Parallelized integration tests for 1.73x speedup
  - Integration tests now run scenarios in parallel using xargs
  - Total integration test time reduced from ~45s to ~26s
  - All 27 scenarios still validate correctly
- **CLI Cleanup** - Removed legacy CLI options
  - Removed: `--radius`, `--ior`, `--scale`, `--center` (replaced by `--objects` syntax)
  - All examples and documentation updated to use modern `--objects` syntax
  - Backward compatibility: old options show migration message

### Fixed
- **Shadow Ray Direction** - Corrected directional light direction convention for shadow rays
  - Root cause: Commit bb92d30 incorrectly negated light direction, causing shadow rays to trace away from light sources
  - Shadow rays now correctly trace toward light sources, restoring shadow functionality
  - Established convention: `light.direction` points TO the light source (where light comes from)
  - All 26 shadow tests pass, all 87 integration tests pass
  - Documentation updated across Light.scala, USER_GUIDE.md, OptiXData.h, MengerCLIOptions.scala, helpers.cu
  - Reference images regenerated with corrected lighting
- **Parchment Material** - Corrected to be translucent instead of refractive
  - Changed `opacity` from 0.7 to 1.0 (fully opaque)
  - Parchment now behaves as a glowing matte material, not semi-transparent
- **CI Pipeline** - Fixed PushToGithub job missing git history
  - Added `GIT_DEPTH: 0` to fetch complete repository history
  - Ensures GitHub mirror has full commit history for proper release notes
- **Code Quality CI** - Resolved Docker API version mismatch
  - Updated code_quality job to use compatible Docker API version
  - CI pipeline now runs successfully without Docker socket errors

### Documentation
- **Release Checklist** - Added comprehensive release workflow documentation
  - Complete step-by-step guide from preparation to verification
  - Covers version management, testing, CI pipeline, and post-release checks
  - Documents TEST FAILURE PROTOCOL and common issues
  - Available via `/release-checklist` skill

## [0.4.2] - 2026-01-26

### Added
- **4D Menger Sponges (TesseractSponge)** - Fractal 4D geometry rendering
  - `--objects type=tesseract-sponge:level=N` for volume-based 4D Menger sponge (24 × 48^level faces)
  - `--objects type=tesseract-sponge-2:level=N` for surface-based 4D Menger sponge (24 × 16^level faces)
  - Fractional level support (e.g., `level=1.5` truncates to integer level)
  - Level parameter required and must be non-negative
  - Full 4D projection support (rot-xw, rot-yw, rot-zw, eye-w, screen-w)
  - Material support (glass, chrome, etc.) on projected sponge faces
  - Cylindrical edge rendering with `edge-material` and `edge-radius` parameters
- **Generalized 4D Projection Pipeline** - `Mesh4DProjection` class
  - Refactored `TesseractMesh` to accept any `Mesh4D` instance (not just Tesseract)
  - Backward-compatible `TesseractMesh` factory object preserves existing API
  - `TesseractSpongeMesh` and `TesseractSponge2Mesh` factories for convenient sponge creation
  - All 4D meshes now share the same projection, rotation, and translation logic
- **4D Sponge Type System** - Classification and validation
  - Extended `ObjectType` with `tesseract-sponge` and `tesseract-sponge-2`
  - New `ObjectType.is4DSponge()` helper method
  - Both types classified as hypercubes via `ObjectType.isHypercube()`
  - Validation enforces level requirement for 4D sponges in `ObjectSpec`
- **Performance Warnings** - Automatic threshold checks for high-level sponges
  - tesseract-sponge: warns at level ≥2 (55K faces), errors at level >4 (127M faces)
  - tesseract-sponge-2: warns at level ≥3 (98K faces), errors at level >5 (25M faces)
  - Estimated triangle counts logged to inform users of render complexity
  - Warnings are advisory only - no hard rejection of high levels
- **Edge Rendering for All 4D Types** - Generalized cylinder edge extraction
  - `TesseractEdgeSceneBuilder` supports tesseract, tesseract-sponge, tesseract-sponge-2
  - Dynamic edge extraction from any `Mesh4D` (not limited to 32 edges)
  - Edge count grows with sponge level (e.g., level 1 sponge has ~1,152 edges)
  - Instance budget calculation accounts for variable edge counts

### Changed
- **MeshFactory** - Added 4D sponge cases
  - `tesseract-sponge` and `tesseract-sponge-2` now supported in `MeshFactory.create()`
  - Both use 4D projection parameters from `ObjectSpec.projection4D`
- **OptiX Engine** - Integrated performance warnings
  - `warnIfHighLevel()` now handles 4D sponge types with triangle estimates
- **CLI Help** - Updated `--objects` description
  - Clarified level requirement for 4D sponges: `level=L (required)`
  - Generalized 4D parameter descriptions

### Technical Details
- **Architecture**: All 4D meshes (Tesseract, TesseractSponge, TesseractSponge2) implement `Mesh4D` trait
- **Projection**: 4D faces → 3D quads → 2 triangles per quad
- **Edge Extraction**: Canonical ordering deduplicates edges from quad faces
- **Backward Compatibility**: Existing `TesseractMesh` usage unaffected

## [0.4.2] - 2026-01-26

### Added
- **Tesseract (4D Hypercube)** - Render 4D geometry projected to 3D via OptiX
  - `--objects type=tesseract` for 4D hypercube rendering (16 vertices, 24 faces projected to 3D)
  - 4D projection parameters: `eye-w=W`, `screen-w=W` (default: 3.0, 1.5)
  - 4D rotation parameters: `rot-xw=DEG`, `rot-yw=DEG`, `rot-zw=DEG` (default: 15°, 10°, 0°)
  - Full material support (glass, chrome, etc.) on tesseract faces
  - `TesseractMesh` class for 4D→3D projection with proper normals and UVs
- **Cylinder Primitive** - Custom OptiX primitive for edge rendering
  - Analytical ray-cylinder intersection in CUDA shader
  - Support for 32 cylindrical edges per tesseract
  - `addCylinderInstance()` API for cylinder instances with endpoints and radius
  - CLI: `--objects type=tesseract:edge-material=chrome:edge-radius=0.02`
- **Metallic Reflection on Cylinder Edges** - Single-bounce PBR reflection
  - Cylinder shader uses `handleMetallicOpaque()` for depth 0 metallic materials
  - Diffuse fallback for depth > 0 to prevent stack overflow
  - Stack size increased from 32KB to 48KB for metallic cylinder rendering
  - Chrome and copper edges show realistic mirror-like reflections
- **Interactive 4D Rotation** - Mouse-based manipulation of 4D objects
  - Left-drag: XW plane rotation (horizontal movement controls 4D rotation)
  - Right-drag: YW plane rotation (horizontal movement controls 4D rotation)
  - Middle-drag: ZW plane rotation (horizontal movement controls 4D rotation)
  - Vertical movement on all drags controls 3D camera pitch
  - Camera state preserved during 4D rotation (position, target, up vector)
- **Edge Material Properties** - Separate materials for tesseract edges
  - `edge-material=PRESET` for preset materials on edges (chrome, copper, glass, etc.)
  - `edge-color=#RRGGBB` for custom edge colors
  - `edge-emission=VALUE` for glowing edges (0.0-1.0)
  - `edge-radius=VALUE` for cylinder thickness (default: 0.02)
- **Emission Property** - Self-illuminating materials
  - Added `emission` field to Material case class (0.0-1.0)
  - Emissive materials glow without requiring light sources
  - Film and Parchment preset materials with emission values
- **Headless Rendering** - Batch processing without window display
  - `--headless` flag renders directly to file without displaying window
  - Invisible window creation using LibGDX's `setInitialVisible(false)`
  - Requires `--save-name` to be specified
  - Useful for CI/CD, batch processing, and remote servers
- **Scene Builder Architecture** - Strategy pattern for object type handling
  - `SceneBuilder` trait with validate/buildScene/calculateInstanceCount methods
  - `SphereSceneBuilder` for pure sphere scenes
  - `TriangleMeshSceneBuilder` for cubes and sponges
  - `CubeSpongeSceneBuilder` for GPU-instanced cube sponges
  - `TesseractEdgeSceneBuilder` for tesseracts with cylindrical edges
  - Automatic builder selection based on object types
  - Validation prevents incompatible object combinations
- **Input Abstraction Layer** - Clean separation of LibGDX and rendering logic
  - `InputEvent` ADT for key/mouse events (KeyPress, KeyRelease, MouseDrag)
  - `InputHandler` trait for processing input events
  - `GdxKeyHandler` and `OptiXKeyHandler` for specific handling
  - `GdxCameraHandler` and `OptiXCameraHandler` for camera manipulation
  - `LibGDXInputAdapter` bridges LibGDX callbacks to event system
  - Zero LibGDX dependencies in camera/key handler logic
- **User Guide** - Comprehensive documentation (1630 lines)
  - Quick start guide with installation and first render
  - Basic usage: spheres, cubes, sponges with materials
  - 4D visualization guide: tesseracts, rotation, projection
  - Headless rendering for batch processing
  - Advanced topics: multiple objects, custom lighting, performance
  - Examples gallery with render commands
- **Projection4DSpec** - 4D projection parameter encapsulation
  - Separate case class for 4D-specific parameters (eyeW, screenW, rotations)
  - Default values defined in companion object
  - Used by ObjectSpec for tesseract configuration

### Changed
- OptiX pipeline now includes cylinder custom primitive hit groups
- Stack size increased from 32KB to 48KB for metallic cylinder shaders
- Shader file renamed from `sphere_combined.cu` to `optix_shaders.cu`
- Input handling refactored to use event-based architecture instead of controller pattern
- Camera manipulation extracted to separate handler classes
- Scene building logic extracted from OptiXEngine to dedicated builder classes
- Material extraction logic moved to `MaterialExtractor` utility
- Texture loading logic moved to `TextureManager` utility

### Fixed
- Screenshot vertical flip bug - images now saved with correct orientation
  - Added `flipVertically()` method in ScreenshotFactory
  - OpenGL framebuffer (bottom-left origin) correctly converted to PNG (top-left origin)
- Infinite pipeline rebuild loop for tesseract edge rendering
  - Fixed by properly tracking pipeline state in TesseractEdgeSceneBuilder
- Cylinder module cleanup causing double-free crash
  - Fixed GAS buffer management for cylinder primitives
- Crash when rotating tesseract with chrome edges
  - Resolved by implementing single-bounce reflection strategy

### Tests
- All 1159 tests passing (394 in menger-app, 765 in optix-jni + C++)
- Test coverage: 82.04% (up from 78.01%)
- 51 integration tests passing (basic objects, multi-object, materials, tesseract, headless)
- New test suites:
  - `TesseractMeshSuite` - 18 tests for 4D→3D projection
  - `TesseractIntegrationSuite` - 25 tests for tesseract rendering pipeline
  - `CylinderSuite` - 34 tests for cylinder primitive and intersection
  - `Camera4DRotationSuite` - 21 tests for interactive 4D rotation
  - `InputEventSuite` - 11 tests for input event system
  - `KeyHandlerSuite` - 12 tests for key event handling
  - `CameraHandlerSuite` - 6 tests for camera manipulation

## [0.4.1] - 2026-01-12

### Added
- **Material System** - PBR-based material properties for realistic rendering
  - Material case class with baseColor, metallic, roughness, IOR, alpha
  - Material presets: glass, water, diamond, chrome, gold, copper, metal, plastic, matte
  - Per-object material assignment via CLI: `--objects type=sphere:material=glass`
  - Per-object color override: `--objects type=cube:material=chrome:color=#FF0000`
- **UV Coordinates** - Texture mapping foundation
  - 8-float vertex format: position (3) + normal (3) + UV (2)
  - UV generation for cube and sponge meshes
  - Box mapping for procedural UV assignment
- **Texture Support** - Image-based surface coloring
  - `TextureLoader` utility for PNG/JPEG loading
  - `--texture-dir` CLI option for texture search path
  - Per-object texture assignment: `--objects type=cube:texture=checker.png`
  - Texture sampling in shaders with bilinear filtering
  - Per-instance texture indices in IAS mode
- **Multi-Project Build Structure** - Reorganized into modular subprojects
  - Separate modules: common, mengerApp, native
  - Improved build isolation and dependency management
- **Comprehensive Error Handling** - Robust error reporting and validation
  - Specific exception types: InvalidMaterialException, TextureLoadException, InvalidObjectSpecException
  - Detailed error context with actionable messages
  - Material and object specification validation
- **Test Coverage Improvements** - ~170 new tests for robustness
  - Property-based tests for animation parameters
  - Edge case tests for CLI parsing and object specifications
  - Coverage protection with ratchet mechanism (75.87% threshold)
- **Manual Test Script** - Comprehensive visual regression testing tool
  - Tests materials, textures, multi-object scenes, shadows, and reflections
- **Strategic Debug Logging** - Configurable diagnostic output for troubleshooting

### Changed
- OptiX shaders extended with texture sampling functions
- Vertex format updated from 6 to 8 floats per vertex
- Removed legacy `--object` CLI option (superseded by `--objects`)
- Reduced cognitive complexity across shaders and parsing code
- Extracted shared helper functions and regex patterns to common module

### Fixed
- Reflection formula in `traceReflectedRay` for accurate glass rendering
- Working directory for `sbt run` now correctly uses project root
- Screenshot path handling in ScreenshotFactory

## [0.4.0] - 2026-01-05

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


[0.4.3]: https://gitlab.com/lilacashes/menger/-/compare/0.4.2...0.4.3
[0.4.2]: https://gitlab.com/lilacashes/menger/-/compare/0.4.1...0.4.2
[0.4.1]: https://gitlab.com/lilacashes/menger/-/compare/0.4.0...0.4.1
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
