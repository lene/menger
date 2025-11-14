# CLAUDE.md

Guidance for Claude Code when working with this repository.

## ⚠️ CRITICAL: Alpha Channel Convention

**STANDARD GRAPHICS ALPHA (never confuse this):**
- **alpha = 0.0** → **FULLY TRANSPARENT** (no opacity, no absorption)
- **alpha = 1.0** → **FULLY OPAQUE** (full opacity, maximum absorption)

Applies to: OptiX shaders (`sphere_combined.cu`), Beer-Lambert absorption, Scala Color objects, all tests.

## Active Development

**Current:** Sprint 1 - Implementing Shadow Rays (Feature 1.2) 🔄
**Branch:** `optix-sphere-from-sdk`
**Local Tests:** 95 tests passing (menger + optix-jni)
**Plan:** [optix-jni/SHADOW_RAYS_PLAN.md](optix-jni/SHADOW_RAYS_PLAN.md)
**Progress:** [optix-jni/ENHANCEMENT_PLAN.md](optix-jni/ENHANCEMENT_PLAN.md)

## Project Overview

Scala 3 fractal renderer using LibGDX. Generates Menger sponges (3D), tesseract sponges (4D) via surface subdivision. Supports interactive exploration, animations, fractional levels with alpha blending.

**Components:**
- **Core** (`menger`): LibGDX-based renderer
- **OptiX JNI** (`optix-jni`): CUDA/OptiX GPU ray tracing

## Build Requirements

**Required:** CUDA 12.0+, OptiX SDK 9.0+, CMake 3.18+, C++17 compiler

**OptiX SDK version MUST match driver:**
- Driver 580.x+ → OptiX SDK 9.0+
- Driver 535-575.x → OptiX SDK 8.0
- Check: `strings /usr/lib/x86_64-linux-gnu/libnvoptix.so.* | grep "OptiX Version"`

## Development Commands

```bash
sbt compile              # Compile (includes C++/CUDA)
sbt test --warn          # Run all tests
sbt "testOnly ClassName" # Run specific test
sbt run                  # Run application
sbt "scalafix --check"   # Check code quality

# OptiX JNI specific
sbt "project optixJni" nativeCompile
rm -rf optix-jni/target/native && sbt "project optixJni" compile  # Clean rebuild
```

## Code Quality Rules

**Wartremover:** No `var`, `while`, `asInstanceOf`, `throw`
**Scalafix:** OrganizeImports, DisableSyntax (noNulls, noReturns), no unused imports
**Style:** Max 100 chars/line, functional style, descriptive names over docstrings
**No null:** Use `Option`, `Try`, or `Either` instead of null - null is forbidden in all Scala code

## Development Checklist

1. Update CHANGELOG.md (keepachangelog.com format)
2. Never `git add -A` - add files explicitly
3. Run `sbt compile && sbt test --warn` before committing
4. Run `sbt "scalafix --check"`
5. Run pre-push hook before pushing
6. Monitor pipeline after pushing

## Common Issues

**CUDA error 718:** OptiX SDK/driver version mismatch. Reinstall matching SDK, `rm -rf optix-jni/target/native`, rebuild.

**PTX file not found after `sbt clean`:**
```bash
mkdir -p target/native/x86_64-linux/bin
cp optix-jni/target/classes/native/x86_64-linux/sphere_combined.ptx target/native/x86_64-linux/bin/
```

**Permission errors after Docker builds:** `pkexec chown -R $USER:$USER optix-jni/target/`

**More troubleshooting:** See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Code architecture, object hierarchy, rendering pipeline
- **[docs/PHYSICS.md](docs/PHYSICS.md)** - OptiX physics (Fresnel, Beer-Lambert, Snell's law)
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Complete troubleshooting guide
- **[docs/CI_CD.md](docs/CI_CD.md)** - GitLab runner setup, Docker images, CI debugging
- **[optix-jni/README.md](optix-jni/README.md)** - OptiX JNI architecture details
- **[docs/GPU_DEVELOPMENT.md](docs/GPU_DEVELOPMENT.md)** - AWS EC2 GPU development setup

## Project Conventions

- Scala 3 style always (never Scala 2)
- AnyFlatSpec for tests
- One import per line
- Functional style: avoid mutable state, exceptions (use Try/Either)
- No docstrings - use descriptive function/parameter names
- Comments only for domain-specific problems not expressible in code
- Always run pre-push hook to catch errors early
