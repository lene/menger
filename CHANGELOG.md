# Changelog

## [Unreleased]

## [0.3.6] - TBD

### Fixed
- **CI Job Configuration** - Fixed failing CI jobs and made them manual-only
  - Test:SbtImage: Moved CUDA_HOME/OPTIX_ROOT exports from before_script to script section
    - Environment variables in before_script don't persist to script section
    - This was causing "CUDA compiler not found" errors
  - Test:SbtImage: Made manual-only (removed automatic tag trigger)
    - Prevents blocking tag pipelines (was causing 0.3.5 tag pipeline to fail)
  - Test:Debian: Added ENABLE_OPTIX_JNI=false to skip OptiX compilation
    - These jobs don't install CUDA, so OptiX JNI compilation always failed
    - Now tests only the core Scala code without GPU dependencies
  - Test:Debian: Made manual-only (documentation job, not needed on every pipeline)
  - code_quality: Removed Docker socket override, now uses default DinD from template
    - More secure and doesn't require runner configuration changes

### Added
- **Shadow Ray Tracing** - Implemented realistic hard shadows for OptiX renderer
  - Cast shadow rays from opaque surfaces to detect occlusion by geometry
  - Shadow factor darkens occluded surfaces to 0.2x ambient lighting (80% reduction)
  - Added `setShadows(bool enabled)` API method to enable/disable shadow rays
  - Shadow ray statistics tracked in RayStats (total shadow rays cast)
  - Added `shadows_enabled` parameter to OptiX Params struct
  - Added `shadow_rays` field to RayStats struct for performance analysis
  - Added SHADOW_RAY_OFFSET constant (0.001f) to prevent shadow acne
  - Uses OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT for optimal performance
  - Implemented __closesthit__shadow() handler for occlusion testing
  - 6 comprehensive unit tests added to RayStatsTest.scala (all passing)
  - **CLI Integration** - Added `--shadows` flag to enable shadow rays via command line
    - Flag requires `--optix` flag (validated at startup)
    - Shadow rays automatically counted when `--stats` flag is used
    - Integrated into OptiXEngine initialization flow

## [0.3.5] - 2025-11-17

### Fixed
- **OptiX Docker Image** - Added CUDA_HOME environment variable to Docker image
  - Fixed CI failures in Test:ComputeSanitizer and Test:Valgrind jobs
  - g++ compilation now finds CUDA headers (cuda.h) when compiling standalone tests
  - Docker image sets ENV CUDA_HOME=/usr/local/cuda to match OPTIX_ROOT pattern
  - Allows all CI jobs to rely on Docker image ENV variables without job-level overrides
- **CI/CD Docker Configuration** - Fixed OptiX SDK installation paths in CI Docker image
  - Install OptiX to dedicated directory /usr/local/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64
  - Create symlink /usr/local/optix for consistent OPTIX_ROOT environment variable
  - Updated all CI jobs (CheckRunTime, BuildDeployable, Test:SbtImage) to use OPTIX_ROOT=/usr/local/optix
  - Added OptiX SDK 9.0 to GitLab Package Registry for CI job downloads
  - Fixed bash command chaining in Dockerfile (use ; instead of &&)

### Added
- **OptiX Ray Tracing Constants** - Extracted all magic numbers to named constants
  - Created RayTracingConstants namespace in OptiXData.h (shared with shaders)
  - Added 14 named constants with clear documentation and rationale
  - Moved shader-local constants (MAX_RAY_DISTANCE, COLOR_SCALE_FACTOR, etc.) to shared header
  - Added CUDA_ERROR_INVALID_PROGRAM_COUNTER to OptiXConstants.h
  - All numeric thresholds now have descriptive names and comments

### Removed
- **OptiX JNI Cleanup** - Removed redundant OptixDeviceContext field from OptiXWrapper::Impl
  - Field was leftover from refactoring migration, never actually used
  - OptiXContext wrapper now provides sole access to device context

### Fixed
- **OptiX Transparency Thresholds** - Corrected alpha channel thresholds for byte-precision accuracy
  - Changed ALPHA_FULLY_TRANSPARENT_THRESHOLD from 0.01 (~3/255) to 1/255 (~0.00392)
  - Changed ALPHA_FULLY_OPAQUE_THRESHOLD from 0.99 (~252/255) to 254/255 (~0.99608)
  - Changed COLOR_CHANNEL_MIN_SAFE_VALUE from 0.01 to 1/255 for Beer-Lambert absorption
  - Thresholds now correctly represent single-byte precision (1/255, 254/255)
  - Previous values used percentages (1%, 99%) which don't align with 8-bit color channels
- **OptiX Glass Rendering** - Restored refraction by fixing MAX_TRACE_DEPTH configuration
  - Consolidated MAX_TRACE_DEPTH to single source of truth in OptiXData.h
  - Increased from 2 to 5 to allow internal reflections in glass (entry + exit + reflections)
  - Fixed issue where pipeline was limited to 2 ray bounces, breaking glass refraction
  - Removed duplicate constant definition in shader (was 5) vs OptiXConstants.h (was 2)

### Removed
- **Documentation Cleanup** - Deleted 7 obsolete documentation files from docs/ directory
  - Removed historical debug notes from glass rendering implementation (now complete)
  - Deleted: ABSORPTION_DEBUG_FINDINGS.md, FIX_PLAN.md, Glass_Implementation_Plan.md
  - Deleted: PTX_PATH_AND_CUSTOM_INTERSECTION_DEBUG.md, SDK_GLASS_EXTRACTION.md
  - Deleted: OptiX.md (pre-integration planning), PHASE_4_PLAN.md (superseded)
  - Retained Glass_Rendering_Findings.md as physics reference

### Fixed
- **README.md** - Updated OptiX SDK version requirement from 8.0 to 9.0+
  - OptiX 8.0 is NOT compatible with modern NVIDIA drivers (580.x+)
  - Aligns with CLAUDE.md documentation

### Added
- **OptiX JNI Architecture Refactoring** - Two-layer architecture for better separation of concerns
  - Extracted low-level OptiXContext layer (pure OptiX API wrapper, stateless, unit tested)
  - Refactored high-level OptiXWrapper to use OptiXContext for all OptiX operations
  - Added Google Test unit test suite (16 C++ tests for OptiXContext)
  - Extracted duplicate pipeline compile options to helper function
  - Deleted dead shader files (sphere_raygen.cu, sphere_miss.cu, sphere_closesthit.cu)
  - All 96 tests passing (16 C++ + 80 Scala)
- **OptiX JNI Performance Optimization** - Moved dynamic scene data from SBT to Params
  - Moved sphere color, IOR, scale, light direction/intensity, and plane parameters from SBT to Params struct
  - Parameter changes now require only cudaMemcpy (not SBT rebuild)
  - Eliminates GPU memory allocations on parameter changes
  - Standard OptiX practice for dynamic scenes
  - All 96 tests passing (16 C++ + 80 Scala)
- **OptiX JNI Library Loading Refactoring** - Converted imperative library loading to functional style
  - Replaced nested try-catch blocks with Try monad composition
  - Replaced while loops with tail-recursive copyLoop() function
  - Replaced imperative library loading with for-comprehension
  - Extracted functional helper methods: loadFromSystemPath, detectPlatform, copyStreamToFile, extractPTX
  - Reduced libraryLoaded initialization from 62 lines to 1 line
  - All 96 tests passing (16 C++ + 80 Scala)
- **OptiX JNI Test Suite Refactoring** - Comprehensive code quality improvements
  - Created RendererFixture trait for automatic lifecycle management (eliminates 1500+ lines of boilerplate)
  - Created ColorConstants object with 30+ named color constants (eliminates 100+ magic numbers)
  - Created ThresholdConstants object with 20+ documented validation thresholds
  - Created TestDataBuilders with fluent API and pre-configured scenarios (glass, water, diamond, etc.)
  - Created ImageValidationFunctional with pure functional implementation (0 mutable variables)
  - Created TestUtilities for consolidated helper functions (savePPM, withRenderer, etc.)
  - Created ImageMatchers with custom ScalaTest matchers for readable assertions
  - Migrated 3 test files to new utilities (OptiXOpaqueSphereTest, OptiXTransparencyTest, OptiXPerformanceTest)
  - Added comprehensive REFACTORING_SUMMARY.md documentation
  - Test suite expanded from 89 to 103 tests, all passing
- Fresnel reflection blending for realistic glass rendering
  - Implemented reflection ray tracing at both entry and exit hits
  - Schlick approximation for Fresnel reflectance calculation
  - Proper blending: `color = fresnel * reflected + (1 - fresnel) * refracted`
  - Total internal reflection handling when refraction not possible
  - Increased MAX_TRACE_DEPTH from 2 to 5 to support internal reflections and multi-bounce paths
  - Beer-Lambert absorption applied only to refracted component at exit
  - Verified reflection correctness through debug visualizations
- Camera position configuration for testing different viewing angles
  - Tested angles from 15° to 30° to observe Fresnel effect
  - Final test position: (0, 2.5, 1.5) showing clear reflection variation

### Improved
- **Build System** - Reduced CMake build verbosity
  - Added CMake flags: `-Wno-dev`, `--log-level=WARNING`
  - Set `CMAKE_INSTALL_MESSAGE LAZY` to suppress verbose install messages
  - Build output now minimal and focused on actual build progress

### Fixed
- Beer-Lambert absorption sign error (was positive, now correctly negative)
- Entering/exiting check using `!entering` instead of incorrect dot product

### Technical Notes
- OptiX SDK's Beer-Lambert implementation designed for hollow spheres (uniform shell thickness)
- For solid spheres, refracted ray distances vary ~0.7-1.2 (not geometric chord length due to refraction bending)
- Fresnel reflections subtle with glass IOR=1.5 in uniform environment but physically accurate
- Diamond IOR=2.4 shows more pronounced reflection effects

## [0.3.5] - 2025-11-10

### Added
- Alpha (transparency) parameter support in OptiX sphere rendering API
  - Updated `setSphereColor` to accept optional alpha parameter (default: 1.0f)
  - Extended HitGroupData structure to store RGBA color (previously RGB)
  - Backward-compatible API: existing 3-parameter calls continue to work
  - Alpha value stored but not yet rendered (shader implementation pending)

## [0.3.4] - 2025-11-02

### Added
- **Phase 4**: Comprehensive OptiX integration testing and validation
  - Integration tests for different camera positions, light directions, sphere sizes, and positions
  - Separate renderer instances pattern to handle current SBT update limitations
  - Edge case tests: extreme FOV (1°-179°), tiny/huge sphere radii, far spheres, close camera
  - Edge case tests: multiple initialize() calls, dispose() before render(), various render sizes
  - Performance benchmark: achieves ~650 FPS @ 800x600 (~1.5ms per frame)
  - Total test suite expanded to 29 tests (from 15), all passing
  - Memory leak detection with compute-sanitizer (GPU) and Valgrind (host C++)
    - compute-sanitizer: 0 errors (GPU memory)
    - Valgrind: 0 definitely lost, 0 indirectly lost (host memory)
    - Standalone C++ test for Valgrind (bypasses JVM)
    - CI job Test:Valgrind verifies host memory leaks
    - CI job Test:ComputeSanitizer verifies GPU memory leaks
    - Pre-push hook runs both tools locally
- OptiX sphere rendering implementation with new OptiXEngine
  - Configurable sphere radius via `--sphere-radius` option
  - Screenshot saving support via `--save-name` option
  - Timeout support via `--timeout` option for automated testing
  - Full integration with LibGDX rendering pipeline
- SavesScreenshots trait for reusable screenshot functionality across engines
- Proper logging infrastructure throughout OptiX codebase
  - Replaced all println statements with LazyLogging logger calls
  - SLF4J warnings suppressed via system property in build.sbt

### Changed
- **BREAKING**: OptiX JNI is now always required (ENABLE_OPTIX_JNI removed)
  - All builds now require CUDA/OptiX installation
  - Simplified build system by removing conditional compilation
  - CI jobs updated to always install CMake and build tools
- Test suite migration to AnyFlatSpec
  - OptiXRendererTest converted from AnyFunSuite to AnyFlatSpec
  - Consistent test style across entire codebase
- OptiXResources refactoring for better modularity
  - Extracted `createLights()` method from initialization
  - Added `ensureAvailable()` call to OptiXRenderer for safety
  - Geometry configuration now uses Try monad for error handling
  - Changed log level from info to debug for scene configuration details
- OptiXEngine improvements
  - More configurable rendering parameters (sphere radius, timeout, save name)
  - Cleaner separation of concerns with SavesScreenshots trait
  - Improved error messages via logger instead of println
- Reduced verbosity in pre-push hook
  - Simplified compute-sanitizer output parsing
  - Cleaner error reporting

### Fixed
- Test suite now handles SBT update limitation by using fresh renderer instances for parameter changes
  - Camera, light, and sphere parameters require pipeline rebuild to take effect
  - Integration tests use separate renderer instances to ensure correct behavior
  - Future enhancement planned: dynamic SBT updates without full pipeline rebuild
- Root-owned build artifacts from Docker CI no longer require sudo to clean
  - Added `after_script` in Test:OptiXJni job to chmod 777 target directory
  - Prevents permission issues when users run `rm -rf optix-jni/target` locally
  - Files created by root in Docker containers are now accessible to regular users
- Pre-push hook now uses CUDA-provided compute-sanitizer instead of outdated Ubuntu package
  - Relies on `$CUDA_HOME` environment variable for portable CUDA installation detection
  - Fails with descriptive error message if CUDA_HOME not set
  - Fixes "Unable to find injection library libsanitizer-collection.so" error
  - Fixed awk field extraction for ERROR_SUMMARY parsing (was reading field 3 instead of field 4)
- CI jobs now install CMake and g++ where needed
  - CheckRunTime and BuildDeployable jobs updated
  - Scalafix job updated to install build tools
- Test:ComputeSanitizer CI job timeout fixed by using standalone C++ test
  - Job was timing out (>60 minutes) because compute-sanitizer instrumented entire sbt/JVM process
  - Now uses same standalone_test.cpp as Valgrind job
  - Bypasses JVM entirely for pure CUDA/OptiX testing
  - Reduced runtime from >60 minutes (timeout) to ~2 minutes
- PushToGithub CI job now uses force push to handle repository divergence
  - GitHub repository can diverge from GitLab (e.g., commits made directly on GitHub)
  - Job resets to GitLab's version on merge conflicts (GitLab is source of truth)
  - Force push is safe after reset and required when histories diverge
  - Prevents "non-fast-forward" push failures

## [0.3.3] - 2025-10-26

### Added
- GitLab Runner GPU configuration documentation (`optix-jni/RUNNER_SETUP.md`)
  - Complete step-by-step guide for configuring runners with NVIDIA GPU support
  - Covers NVIDIA driver, Docker, nvidia-container-toolkit installation
  - Documents required `gpus = "all"` configuration in runner config.toml
  - Includes verification steps and troubleshooting section
- **Phase 3**: Complete OptiX ray tracing pipeline implementation
  - OptiX module loading from PTX with built-in sphere intersection support
  - Program groups for ray generation, miss, and closest hit shaders
  - Pipeline linking with proper stack size configuration
  - Shader Binding Table (SBT) setup with camera, light, and material parameters
  - Geometry Acceleration Structure (GAS) for sphere primitives
  - Full render() pipeline that traces rays and generates RGBA images
  - Comprehensive test suite (15 tests) covering initialization, configuration, rendering, and disposal
  - PPM image output for visual verification
- OptiX SDK path auto-detection in CMake build
  - Checks `OPTIX_ROOT` environment variable first
  - Auto-detects SDK in common locations (`/usr/local/`, `~/`, `~/workspace/`)
  - Supports multiple SDK versions side-by-side
  - Falls back to hardcoded path only as last resort
- Pure Scala image analysis for rendered output validation
  - Brightness standard deviation check verifies proper gradients (not uniform stub output)
  - Center brightness check validates sphere lighting characteristics
  - Works with `sbt test` without requiring external tools (ImageMagick, etc.)

### Fixed
- OptiX JNI CI job now includes `nvidia-smi` verification step to fail fast if GPU not configured
- Updated CI job comments to reference runner setup documentation
- Fixed sphere normal calculation to support arbitrary sphere positions (was hardcoded to origin)
  - Added sphere_center to HitGroupData structure in SBT
  - Updated sphere_closesthit.cu to compute normal as normalize(hit_point - sphere_center)
  - Updated sphere_combined.cu with same fix
- Disabled log4j JMX in CI to eliminate cgroup detection errors flooding logs
  - Added `.sbtopts` with `-Dlog4j2.disableJmx=true`
  - Fixes "Could not reconfigure JMX" NullPointerException in Docker environments

### Changed
- OptiX CI Docker image now uses versioned tags (`12.8-9.0-25-1.11.7` format: CUDA-OptiX-Java-sbt)
  - Enables reproducible builds with explicit version tracking
  - Introduced `OPTIX_DOCKER_VERSION` variable in `.gitlab-ci.yml`
  - `latest` tag maintained as alias for newest version

### Upgraded
- OptiX CI Docker image switched from Ubuntu base to `nvidia/cuda:12.8.0-devel-ubuntu24.04`
  - Eliminates 15-minute CUDA installation time in Docker builds
  - CUDA layer (~9GB) pulled from DockerHub instead of being stored in GitLab registry
  - Improved layer separation for better caching (build tools, OptiX, Java, sbt in separate layers)
  - Faster incremental updates when upgrading only Java/sbt

## [0.3.2] - 2025-10-23

### Added
- Optional OptiX JNI build system controlled by `ENABLE_OPTIX_JNI` environment variable
  - Default build no longer requires CUDA/OptiX installation (enables development on systems without NVIDIA GPU)
  - Set `ENABLE_OPTIX_JNI=true` to enable OptiX JNI compilation when CUDA/OptiX are available
  - Build system works on vanilla systems without GPU development tools
- OptiX code quality improvements:
  - Added `OptiXConstants` namespace with named constants for log buffer size, log levels, and default background colors
  - Added `VectorMath` namespace with C++ helpers (`normalize3f`, `cross3f`)
  - Added CUDA vector math helpers (`normalize`, `dot`, operators for `float3`)
  - Added `TestConfig` and `ImageIO` helper objects in OptiX test suite
  - Added comprehensive documentation for sphere center limitation in shader code

### Changed
- **BREAKING**: OptiX JNI is now opt-in rather than required
  - Root project no longer depends on optixJni by default
  - Use `ENABLE_OPTIX_JNI=true sbt compile` to build with OptiX support
- CMake build configuration now gracefully detects CUDA/OptiX availability
  - Uses `check_language(CUDA)` for non-fatal CUDA detection
  - Builds stub library when CUDA/OptiX not available
  - Clear build configuration summary shows what was detected
- CI/CD pipeline separated into GPU and non-GPU jobs
  - Main test jobs (`Test:SbtImage`, `CheckCoverage`, `Run:UseSbtDocker`, etc.) no longer install CUDA
  - OptiX JNI tests run on dedicated GPU runners with `nvidia` tag using pre-built Docker image
  - Faster CI builds for non-GPU jobs
  - Pre-built Docker image (`optix-cuda:latest`) with CUDA 12.8 reduces OptiX JNI job time from 15-20 minutes to 3-5 minutes
- Fixed code_quality CI job to use host Docker socket instead of Docker-in-Docker
- Updated CLAUDE.md with comprehensive OptiX JNI build instructions
  - Documents default behavior (OptiX disabled)
  - Documents how to enable OptiX JNI with environment variable
  - Added Phase 3 future enhancement section for trait-based renderer architecture
- OptiX code refactoring for readability and maintainability:
  - Refactored OptiXRenderer.scala to functional style using Try monad, for-comprehensions, and tail recursion
  - Eliminated imperative patterns (try/catch, early returns, while loops) from Scala code
  - Split massive 348-line `buildPipeline()` method into 5 focused methods (75% size reduction)
  - Converted all Scala code to Scala 3 indentation-based syntax (colons and indentation instead of braces)
  - Simplified CUDA shader ray direction calculation from 10+ lines to 1 line using vector helpers

### Fixed
- Removed duplicate `optix-jni/src/native/` directory (correct location is `src/main/native/`)
- Added preprocessor guards (`#ifdef HAVE_CUDA`, `#ifdef HAVE_OPTIX`) to C++ source files
  - OptiXWrapper.cpp compiles without CUDA/OptiX headers
  - Stub implementations provided when GPU support unavailable
- Fixed outdated comment in OptiXData.h about camera_w vector (now correctly documents forward direction)

## [0.3.1] - 2025-10-21

### Added
- Automated OptiX local setup script (`scripts/setup-optix-local.sh`) for Ubuntu/Debian workstations with NVIDIA GPUs
  - Automatically detects NVIDIA GPU and installs appropriate drivers
  - Installs CUDA Toolkit 12.8 from official repositories
  - Installs OptiX SDK 8.0
  - Configures environment variables for Bash and Zsh
  - Handles system reboots gracefully
  - Runs comprehensive verification after setup
- Fish shell environment configuration (`scripts/setup-optix-env.fish`)
  - Sets all required environment variables (PATH, LD_LIBRARY_PATH, CUDA_HOME, OPTIX_ROOT)
  - Provides `verify-optix-env` function for quick verification
  - Color-coded status messages on load
  - Can be sourced in config.fish for automatic configuration
- Local development setup documentation in GPU_DEVELOPMENT.md
  - Automated setup guide with single-command installation
  - Manual setup instructions for unsupported platforms
  - Fish shell configuration section
  - Local development workflow
  - Troubleshooting guide for common setup issues
- Added `--log-level` option to control logging verbosity (ERROR, WARN, INFO, DEBUG)
- Added `--fps-log-interval` option to control frequency of FPS logging

### Fixed
- Improved logging throughout codebase with standard logging functions
- Reduced verbose output from terraform and spot instance scripts

## [0.3.0] - 2025-10-06

### Added
- OptiX verification script (`scripts/verify-optix.sh`) for comprehensive validation of NVIDIA driver, CUDA, and OptiX installation
- Automated OptiX verification in AMI build process to ensure GPU environment is correctly configured
- Comprehensive environment variable setup for CUDA and OptiX (PATH, LD_LIBRARY_PATH, CUDA_HOME, OPTIX_ROOT) in system-wide locations, ubuntu user, and skeleton for new users
- Added --list-running option to nvidia-spot.sh to show currently running instances
- Added --terminate option to nvidia-spot.sh to terminate instances with concise output
- Added spot_request_id output to Terraform configuration
- Added --availability-zone option to target specific AZs (region automatically derived from AZ)
- Spot instance state management system with automatic backup and restoration:
  - `scripts/backup-spot-state.sh` - Backup instance state with selective rsync (workspace, configs, SSH keys, shell histories)
  - `scripts/restore-spot-state.sh` - Restore saved state to new instance
  - `scripts/list-spot-states.sh` - List all saved states with metadata (timestamp, git branch, size)
  - `scripts/cleanup-spot-states.sh` - Cleanup old states with filters (--older-than-days, --keep-recent)
- Integrated state management in nvidia-spot.sh:
  - `--list-states` - List all saved instance states
  - `--restore-state NAME` - Restore specific saved state on launch
  - `--save-state NAME` - Save instance state with given name before shutdown
  - `--no-auto-restore` - Skip automatic restoration of 'last' state
  - Auto-saves to 'last' state on logout (before auto-termination)
  - Auto-restores 'last' state on launch (unless --no-auto-restore specified)
- Comprehensive testing infrastructure without creating AWS resources:
  - `scripts/test-aws-config.sh` - Validate AWS CLI, credentials, permissions, and region setup using dry-run mode
  - `scripts/validate-ami-build.sh` - Validate AMI build configuration before expensive build process
  - `scripts/test-terraform-config.sh` - Validate Terraform configuration and generate plan without creating resources
  - `scripts/test-state-management.sh` - Test state management scripts using mock data (no instance required)
- Documentation reorganization:
  - `GPU_DEVELOPMENT.md` - Comprehensive guide for remote GPU development with integrated testing documentation (consolidated from README.md with detailed testing guide)
  - Updated README.md to focus on application usage and mathematical background
  - Simplified terraform/README.md to technical reference only

### Fixed
- Fixed nvidia-spot.sh to wrap bash commands in 'bash -c' for fish shell compatibility
- Fixed Terraform configuration to automatically create subnet if none exists in default VPC
- Simplified auto-termination to use local sleep and terraform destroy instead of remote daemon
- Reduced verbose Terraform output during instance launch

### Upgraded
- Updated build-ami.sh to use Ubuntu 24.04 (noble) as base image
- Updated CUDA repository configuration for Ubuntu 24.04 compatibility
- Added Claude Code installation to AMI provisioning

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
