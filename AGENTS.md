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
- **Never commit failing tests** - all tests must pass before commit (run `sbt test`)
- **Never commit test changes without investigation** - follow TEST FAILURE PROTOCOL
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

## TEST FAILURE PROTOCOL

### ⚠️ CRITICAL: When Tests Fail

**NEVER rewrite tests to make them pass without thorough investigation first.**

Failing tests are often catching real bugs. Follow this protocol:

#### 1. **Investigate Root Cause FIRST**

Before changing ANY test code, determine WHY it failed:

```bash
# Run the specific failing test to see exact failure
sbt "testOnly ClassName -- -z \"test name pattern\""

# Check git history of the test
git log --oneline --follow -- path/to/TestFile.scala

# Check recent changes to code under test
git log --oneline --since="1 week ago" -- path/to/implementation/

# Check if the test passed before recent changes
git checkout <previous-commit>
sbt "testOnly ClassName -- -z \"test name\""
git checkout -
```

#### 2. **Determine Correct Behavior**

Ask these questions:

- **Is the test expectation correct?** Review test logic and assertions
- **Did code behavior change intentionally?** Check commit messages
- **Is this catching a regression?** Compare with previous working versions
- **Does visual output match expectations?** Run visual regression tests if applicable

#### 3. **Decision Tree**

```
Test Fails
    ├─> Bug in IMPLEMENTATION
    │   └─> FIX THE CODE, keep test unchanged
    │
    ├─> Bug in TEST (wrong expectation)
    │   └─> Verify with git history that test was ALWAYS wrong
    │       └─> Document WHY test was wrong in commit message
    │       └─> Fix test expectations
    │
    ├─> Intentional behavior change
    │   └─> Verify change is documented and approved
    │       └─> Update test to match NEW correct behavior
    │       └─> Document in commit: "test: Update for behavior change in <commit>"
    │
    └─> Test became flaky/brittle
        └─> Improve test robustness
        └─> Don't just widen tolerances without understanding WHY
```

#### 4. **Document Your Investigation**

When fixing a test, the commit message MUST explain:

```
test: Fix incorrect expectation in ShadowRayTest

Root cause investigation:
- Test expected 0 shadow rays for light direction (0,0,-1)
- Test passed with commit 7d0e2fc (buggy - missing negation)
- Test failed with commit bb92d30 (fixed - restored negation)
- Analysis: Test expectation was wrong, written for buggy behavior

The test assumed light direction (0,0,-1) would not illuminate
sphere front, but correct shader behavior negates direction,
making it (0,0,+1) which DOES illuminate front faces.

Fix: Change light direction to actually point away from sphere.
```

#### 5. **Red Flags - STOP and Ask User**

These situations require user consultation:

- **Multiple tests failing after "simple" refactor** → Likely introduced bug
- **Visual tests fail but unit tests pass** → Visual tests caught rendering bug
- **Integration tests fail but unit tests pass** → System behavior changed
- **Only some similar tests fail** → Likely edge case bug
- **Test has been stable for months** → Respect test's history

### Visual Regression Testing Value

**Lesson from directional light bug:**

- Unit tests PASSED with buggy code (no negation)
- Visual regression tests CAUGHT the bug (42x threshold exceeded)
- Demonstrates: visual tests catch bugs that unit tests miss

Always run visual tests when changing rendering code:

```bash
sbt test  # Includes visual regression tests in IntegrationSuite
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

   **If tests fail:** Follow TEST FAILURE PROTOCOL above (investigate BEFORE fixing tests)

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
| [CODE_IMPROVEMENTS.md](CODE_IMPROVEMENTS.md) | Code quality assessments |
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

---

## RELEASE WORKFLOW

### Overview

Menger uses a **fully automated release pipeline** triggered by version bumps merged to the main branch via merge requests.

**Total time from commit to published release: ~30-40 minutes**

**Process:**
1. Pre-release prep (manual: 15-30 min) → 2. Pre-push validation (automated: 8-10 min) → 3. Create MR (manual: 5-10 min) → 4. Release pipeline (automated: 12-15 min) → 5. Verification (manual: 5-10 min)

### Quick Start

**For detailed step-by-step instructions, invoke:** `/release-checklist` skill

**Quick reference:**

1. **Update version in 3 files:**
   - `menger-app/build.sbt`
   - `.gitlab-ci.yml` (DEPLOYABLE_VERSION)
   - `menger-app/src/main/scala/menger/MengerCLIOptions.scala`

2. **Update documentation:**
   - CHANGELOG.md (keepachangelog.com format)
   - ROADMAP.md (mark sprint complete)
   - CODE_IMPROVEMENTS.md (run quality assessment)
   - arc42 docs (if architecture changed)

3. **Create release branch and merge request:**
   ```bash
   git checkout -b release/vX.Y.Z
   git add [files]
   git commit -m "release: Version X.Y.Z"
   git push origin release/vX.Y.Z
   glab mr create --title "Release vX.Y.Z" --target-branch main
   ```
   Pre-push hook validates automatically (8-10 minutes)

4. **Merge MR after pipeline passes:**
   - GitLab CI: https://gitlab.com/lilacashes/menger/-/pipelines
   - All jobs must pass (green checkmarks)
   - Merge triggers release pipeline automatically

5. **Verify releases:**
   - GitLab: https://gitlab.com/lilacashes/menger/-/releases
   - GitHub: https://github.com/lene/menger/releases
   - Download and test packaged artifact

### Common Release Issues

**Version mismatch errors:**
- Pre-push hook validates versions across 3 files
- Fix all three and commit again

**Tag already exists:**
- If pushing to GitHub after GitLab release: Normal, PushToGithub job handles it
- If genuine duplicate: Increment version number

**Protected branch error:**
- Expected - main branch is protected
- Always use feature/release branches with merge requests
- Never push directly to main

**Pipeline failures:**
- Check pipeline logs for specific job failure
- Most common: test failures, coverage drops, memory leaks
- Follow TEST FAILURE PROTOCOL for test issues
- Never bypass tests for releases

### Pre-Push Validation

The pre-push hook (`.git_hooks/pre-push`) automatically validates:
- Environment (CUDA_HOME, OPTIX_ROOT)
- GitLab CI config syntax
- Version consistency across 3 files
- Full compilation and test suite (1,070 tests)
- Code quality (scalafix)
- Test coverage ratchet (≥80%, max 1% drop)
- Memory leaks (compute-sanitizer, valgrind)
- Integration tests (27 scenarios)

**Manual run:**
```bash
.git_hooks/pre-push
```

### Release Pipeline Stages

**Automatic on main branch merge:**

1. **Build Stage** - All tests and quality checks
2. **CreateRelease** - Creates GitLab release and git tag
3. **Tag Pipeline** - Uploads artifact, mirrors to GitHub, creates GitHub release

**No manual intervention required** - entire process is automated

### Emergency Hotfix

For critical production bugs:

1. Create hotfix branch from main
2. Make minimal fix
3. Run full pre-push hook: `.git_hooks/pre-push`
4. Bump PATCH version (e.g., 0.4.3 → 0.4.4)
5. Update CHANGELOG.md
6. Create MR and merge to main after pipeline passes

**Hotfix criteria:**
- Production is broken or severely degraded
- Security vulnerability
- Data loss risk

Invoke `/release-checklist` skill for detailed emergency hotfix process.

---

## Miscellaneous notes

- do not write images to /tmp. sanitizeFileName would strip the slashes. write to the current folder or a subfolder.
- you can capture screenshots of the rendering window with `scrot` if needed.