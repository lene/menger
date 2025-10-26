# Changelog

## [Unreleased]

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
