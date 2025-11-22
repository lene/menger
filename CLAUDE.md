# CLAUDE.md

Guidance for Claude Code when working with this repository.

## ⚠️ CRITICAL: Alpha Channel Convention

**STANDARD GRAPHICS ALPHA (never confuse this):**
- **alpha = 0.0** → **FULLY TRANSPARENT** (no opacity, no absorption)
- **alpha = 1.0** → **FULLY OPAQUE** (full opacity, maximum absorption)

Applies to: OptiX shaders (`sphere_combined.cu`), Beer-Lambert absorption, Scala Color objects, all tests.

## ⚠️ CRITICAL: Architecture Documentation (arc42)

**Single source of truth:** [docs/arc42/README.md](docs/arc42/README.md)

**Before making architectural decisions:**
1. **Consult arc42** - Check existing decisions in [Section 9](docs/arc42/09-architectural-decisions.md)
2. **Check quality requirements** - Validate against [Section 10](docs/arc42/10-quality-requirements.md)
3. **Review risks** - Consider impacts per [Section 11](docs/arc42/11-risks-and-technical-debt.md)

**After making changes:**
1. **Update arc42** if the change affects architecture, quality requirements, or introduces new risks
2. **Update sprint plans** if implementing a planned feature - record actual metrics
3. **Update baseline values** in Section 10 when establishing new performance baselines

**Documentation must stay current** - outdated documentation is worse than no documentation.

## Active Development

**Current:** Planning Sprint 5 - Triangle Mesh Foundation
**Branch:** `feature/caustics` (Sprint 4 deferred, branch preserved)
**Local Tests:** ~897 passing (21 C++ + ~876 Scala)
**Progress:** [optix-jni/ENHANCEMENT_PLAN.md](optix-jni/ENHANCEMENT_PLAN.md)

**Completed Sprints:**
- ✅ **Sprint 1** - Foundation (Ray Statistics, Shadow Rays)
- ✅ **Sprint 2** - Interactivity (Mouse Camera Control, Multiple Light Sources)
- ✅ **Sprint 3** - Advanced Quality (Adaptive Antialiasing, Unified Color API, Cache Management)

**Deferred:**
- ⏸️ **Sprint 4** - Caustics (algorithm issues, branch preserved for future)

**Upcoming Sprints (Feature Breadth Focus):**
- **Sprint 5** - Triangle Mesh Foundation (OptiX triangle support)
- **Sprint 6** - Cube Primitive (first mesh object)
- **Sprint 7** - Multiple Objects (scene graph)
- **Sprint 8** - Sponge Mesh Export → **v0.5 Milestone**
- **Sprint 9-10** - Object Animation
- **Sprint 11** - Scene Description Language

**Recent Features (v0.3.7):**
- Unified `menger.common.Color` class with factory methods
- Custom plane colors via `--plane-color` flag
- OptiX cache auto-recovery
- Adaptive antialiasing with `--antialiasing`, `--aa-max-depth`, `--aa-threshold`

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
2. **Update arc42 documentation** if change affects architecture, quality, or risks
3. Never `git add -A` - add files explicitly
4. Run `sbt compile && sbt test --warn` before committing
5. Run `sbt "scalafix --check"`
6. Run pre-push hook before pushing
7. Monitor pipeline after pushing

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

### Architecture (arc42) - Single Source of Truth

**Index:** [docs/arc42/README.md](docs/arc42/README.md)

| Section | Purpose | When to Update |
|---------|---------|----------------|
| [05 - Building Blocks](docs/arc42/05-building-block-view.md) | Code architecture | New components/modules |
| [07 - Deployment](docs/arc42/07-deployment-view.md) | CI/CD, GPU setup | Infrastructure changes |
| [08 - Concepts](docs/arc42/08-crosscutting-concepts.md) | Physics, rendering | Algorithm changes |
| [09 - Decisions](docs/arc42/09-architectural-decisions.md) | ADRs | Any architectural decision |
| [10 - Quality](docs/arc42/10-quality-requirements.md) | Performance targets | New baselines established |
| [11 - Risks](docs/arc42/11-risks-and-technical-debt.md) | Technical debt | New debt or risk identified |

### Other Documentation

- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[optix-jni/README.md](optix-jni/README.md)** - OptiX JNI module details
- **[optix-jni/ENHANCEMENT_PLAN.md](optix-jni/ENHANCEMENT_PLAN.md)** - Sprint roadmap

## Project Conventions

- Scala 3 style always (never Scala 2)
- AnyFlatSpec for tests
- One import per line
- Functional style: avoid mutable state, exceptions (use Try/Either)
- No docstrings - use descriptive function/parameter names
- Comments only for domain-specific problems not expressible in code
- Always run pre-push hook to catch errors early
- when running under xvfb-run, set __GL_THREADED_OPTIMIZATIONS=0 to avoid crashes in libnvidia-glcore.so