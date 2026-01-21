# AGENTS.md - Development Guidelines

Guidance for Claude Code and other AI agents when working with this repository.

---

## ⚠️ CRITICAL CONVENTIONS

### Alpha Channel Convention (NEVER confuse this)
- **alpha = 0.0** → **FULLY TRANSPARENT** (no opacity, no absorption)
- **alpha = 1.0** → **FULLY OPAQUE** (full opacity, maximum absorption)
- Applies to: OptiX shaders (`sphere_combined.cu`), Beer-Lambert absorption, Scala Color objects, all tests

### Architecture Documentation (arc42)
- **Single source of truth:** [docs/arc42/README.md](docs/arc42/README.md)
- **Before architectural decisions:** Consult [Section 9](docs/arc42/09-architectural-decisions.md), [Section 10](docs/arc42/10-quality-requirements.md), [Section 11](docs/arc42/11-risks-and-technical-debt.md)
- **After changes:** Update arc42 if affecting architecture, quality requirements, or risks
- **Documentation must stay current** - outdated documentation is worse than no documentation

### Data Safety
- **Never delete data without user permission**
- Always confirm before destructive operations

---

## TOOL LIMITATIONS & WORKAROUNDS

### Bash Tool Issues (CRITICAL)
Your bash tool has known limitations. Follow these workarounds:

**Use `pkexec` instead of `sudo`:**
- `sudo` requires password entry on terminal and doesn't work with the bash tool
- Always use `pkexec` for privileged operations
- Example: `pkexec chown -R $USER:$USER optix-jni/target/`

**Command chaining:**
- ❌ **DO NOT use `&&`** - it makes the bash tool fail
- ✅ **Use `;` instead** - chains commands correctly
- Example: `sbt compile ; sbt test` (not `sbt compile && sbt test`)

**Command substitution:**
- ❌ **DO NOT use `$()`** - it fails with the bash tool
- ✅ **Run substituted commands separately** instead

### Git Workflow
- **Never `git add -A`** - add files explicitly
- **Never commit automatically** - always show diff for user review first
- when fetching, always use the --all --tags options

---

## CODE STANDARDS & CONVENTIONS

### Language & Style
- **Scala 3 only** - never Scala 2 syntax
- **Line length:** max 100 characters
- **Imports:** one per line, organized per .scalafix.conf
- **No null:** Use `Option`, `Try`, or `Either` - null is forbidden in all Scala code
- **No docstrings:** Use descriptive function/parameter names instead
- **Comments:** Only for domain-specific problems not expressible in code
- **No magic numbers:** Introduce named constants with a clear name instead

### Functional Programming (Enforced)
- **Immutability:** Avoid mutable state, use `val` not `var`
- **Error handling:** Use `Try`/`Either`/`Option` instead of exceptions
- **No side effects:** Pure functions preferred

### Code Quality Tools (Enforced)
- **Wartremover:** No `var`, `while`, `asInstanceOf`, `throw` in production code
- **Scalafix:** OrganizeImports, DisableSyntax (noNulls, noReturns), no unused imports
- **Test framework:** AnyFlatSpec for Scala tests

### Editing Conventions
- **Manual edits only:** Never use scripts or sed for bulk code changes (proven unreliable and corrupt code)
- **LibGDX compatibility:** `@SuppressWarnings` allowed for necessary vars in LibGDX integration

### Runtime Environment
- **xvfb-run:** Set `__GL_THREADED_OPTIMIZATIONS=0` to avoid crashes in libnvidia-glcore.so

---

## BUILD & TEST COMMANDS

```bash
# Core commands
sbt compile                          # Compile all modules (includes C++/CUDA)
sbt test                             # Run all tests (~1,070 total: 27 C++ + 1,043 Scala)
sbt "testOnly ClassName"             # Run specific Scala test
sbt run                              # Run application

# Code quality
sbt "scalafix --check"               # Verify code quality rules

# OptiX JNI specific
sbt "project optixJni" nativeCompile # Compile C++/CUDA only
sbt "project optixJni" nativeTest    # Run C++ Google Test suite
rm -rf optix-jni/target/native ; sbt "project optixJni" compile  # Clean rebuild
```

---

## DEVELOPMENT WORKFLOW

### Standard Development Cycle

1. **Make code changes**
   - Follow code standards above
   - Update CHANGELOG.md (keepachangelog.com format)
     - don't document every single commit. instead give a concise, but complete of the features that hae been added and the bugs that  
       have been fixed. but only preexistig bugs, bugs that were introduced during the development of 0.4.1 and have bee fixed do not  
       get mentioned.

   - Update arc42 documentation if affecting architecture, quality, or risks

2. **Test changes**
   ```bash
   sbt compile
   sbt test
   ```

3. **Verify code quality**
   ```bash
   sbt "scalafix --check"
   ```

4. **Run pre-push hook** (can take up to 10 minutes)

5. **Show diff to user**
   - User reviews all changes
   - User decides when to commit

6. **Monitor pipeline** after pushing

---

## BUILD REQUIREMENTS

### Required Software
- CUDA Toolkit 12.0+
- NVIDIA OptiX SDK 9.0+
- CMake 3.18+
- C++17 compiler
- Java 21+
- sbt 1.11+

### OptiX SDK Version Matching (CRITICAL)
OptiX SDK version **MUST match driver**:
- Driver 580.x+ → OptiX SDK 9.0+
- Driver 535-575.x → OptiX SDK 8.0

**Check your setup:**
```bash
strings /usr/lib/x86_64-linux-gnu/libnvoptix.so.* | grep "OptiX Version"
```

---

## PROJECT STRUCTURE

### Project Overview
This project consists of three main capabilities:

1. **OptiX JNI Wrapper** - Java Native Interface bindings for NVIDIA's OptiX ray tracing library
2. **Ray Tracing Renderer** - Scala 3 implementation with LibGDX integration for interactive rendering
3. **3D/4D Visualization Tool** - Supports rendering and exploration of three and four dimensional objects

**Current showcase application:** Menger sponges (3D) and tesseract sponges (4D) generated via surface subdivision, with support for interactive exploration, animations, and fractional levels with alpha blending.

### Components
- **menger-app** - LibGDX-based renderer, CLI, input handling
- **menger-common** - Domain primitives (Color, Vector, Light), constants
- **optix-jni** - CUDA/OptiX GPU ray tracing (JNI bindings)

---

## TROUBLESHOOTING

### Common Issues

**CUDA error 718:**
- Cause: OptiX SDK/driver version mismatch
- Fix: Reinstall matching SDK, `rm -rf optix-jni/target/native`, rebuild

**PTX file not found after `sbt clean`:**
```bash
mkdir -p target/native/x86_64-linux/bin
cp optix-jni/target/classes/native/x86_64-linux/sphere_combined.ptx target/native/x86_64-linux/bin/
```

**Permission errors after Docker builds:**
```bash
pkexec chown -R $USER:$USER optix-jni/target/
```

**More troubleshooting:** See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## DOCUMENTATION

### Primary Documentation
| Document | Purpose |
|----------|---------|
| [docs/arc42/README.md](docs/arc42/README.md) | **Architecture (single source of truth)** |
| [CHANGELOG.md](CHANGELOG.md) | Version history (keepachangelog.com format) |
| [CODE_IMPROVEMENTS.md](CODE_IMPROVEMENTS.md) | Code quality assessments (updated before each push) |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues and solutions |

### arc42 Architecture Sections
| Section | Update When |
|---------|-------------|
| [05 - Building Blocks](docs/arc42/05-building-block-view.md) | New components/modules |
| [07 - Deployment](docs/arc42/07-deployment-view.md) | Infrastructure changes |
| [08 - Concepts](docs/arc42/08-crosscutting-concepts.md) | Algorithm/physics changes |
| [09 - Decisions](docs/arc42/09-architectural-decisions.md) | Any architectural decision |
| [10 - Quality](docs/arc42/10-quality-requirements.md) | New performance baselines |
| [11 - Risks](docs/arc42/11-risks-and-technical-debt.md) | New technical debt |

### Module Documentation
- [optix-jni/README.md](optix-jni/README.md) - OptiX JNI module details
- [optix-jni/ENHANCEMENT_PLAN.md](optix-jni/ENHANCEMENT_PLAN.md) - Sprint roadmap

---

## DEVELOPMENT STATUS

For current sprint status, completed features, and roadmap, see:
- **Sprint roadmap:** [optix-jni/ENHANCEMENT_PLAN.md](optix-jni/ENHANCEMENT_PLAN.md)
- **Recent changes:** [CHANGELOG.md](CHANGELOG.md)
- **Code quality:** [CODE_IMPROVEMENTS.md](CODE_IMPROVEMENTS.md)

**Note:** Development status details are intentionally kept in separate files to avoid this document becoming stale.

## Miscellaneous notes

- do not write images to /tmp. sanitizeFileName would strip the slashes. write to the current folder or a subfolder.
- you can capture screenshots of the rendering window with `scrot` if needed.