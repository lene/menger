# Code Quality Review — 2026-06-29

Sprint 31 close review. 48 files changed (+2704/-342), 20 commits. L-system grammar engine,
3D/4D turtle interpreters, CLI/DSL integration, curve padding fix, CI restructuring, guardrail
hardening.

## Summary

The sprint delivers a professional L-system implementation with robust grammar rewriting,
stochastic rule selection, full turtle alphabet (ABOP standard + 4D extensions), per-segment
material control, and 4D projection. 7 reference images committed (5 3D + 2 4D). 2,356 tests
pass, 0 failures. Three findings below warrant attention — none are release-blocking for 0.7.8.

## Tooling status

- Scalafix: ✅ CI runs it (fixed import ordering in SceneObject.scala during sprint)
- WartRemover: ✅ runs as part of compile
- ArchUnit: ✅ 14+5 active rules; no new violations from L-system code (moved turtles to
  menger.engines.scene to comply with package restrictions)
- clang-tidy: ✅ configured, runs in pre-push + CI
- cppcheck: ✅ configured, runs in pre-push + CI
- Pre-commit: ✅ unified compile+scalafix across all three repos
- Pre-push: ✅ change-aware (HAS_SCALA/HAS_NATIVE/HAS_RENDERING detection)
- CI: ✅ new 'check' stage for fast-fail, GPU jobs deduplicated to MR-only

## Findings

### 1. LSystemTurtle4D.emitRun has no minimum-points guard

**Where**: `LSystemTurtle4D.scala:211`
**Impact**: Low — 4D presets (HilbertCurve4D, Tree4D) use iterative grammars that produce
long runs. But a custom grammar could generate 1-point segments that get rejected by
`addCurveInstance` (requires ≥4 pts). Inconsistent with `LSystemTurtle3D` which requires ≥2 pts.
**Effort**: 15min

**What**: The 4D turtle's `emitRun` only checks `isEmpty`, while the 3D turtle requires
`points.length >= 2`. If a short run reaches the 4D turtle, it produces a curve spec that
`CurveSceneBuilder.padToMinPoints` will pad to 4 points — but padding a 1-point curve produces
a degenerate result (all 4 points identical = invisible curve).

**Suggested direction**: Mirror the 3D turtle's `points.length >= 2` check in the 4D turtle's
`emitRun`. Add a unit test with a grammar that produces single-point runs.

### 2. LSystemTurtle3D and LSystemTurtle4D share ~80% duplicated logic

**Where**: `LSystemTurtle3D.scala` (534 lines), `LSystemTurtle4D.scala` (261 lines)
**Impact**: Medium — stepF, stepTurn, stepPush/stepPop, and the main process loop are the same
algorithm with different dimension types (Vec3 vs Vector[4]). A bug fix applied to one turtle
must be manually replicated in the other (already happened: the Sprint 31 `stepPush`/`stepTurn`
no-emit fix had to be applied separately to both).
**Effort**: 4h

**What**: Both turtles implement the same L-system interpretation algorithm with separate type
parameters. `stepSymbol` dispatch, `process` tail-recursion, `stepF`/`stepFwdNoRecord`/
`stepPush`/`stepPop`/`stepTurn`/`stepTurn180` are structurally identical. The 4D turtle adds
ana-axis rotations (`>`,`<`) and 4D projection in `emitRun`.

**Suggested direction**: Extract a shared `LSystemTurtleBase` trait parameterized by the point
type (Vec3/Vec4), with dimension-specific operations (rotate, project) supplied by the
implementing class. The common processing loop, emit logic, and branch management would live
once. This also fixes finding #1 automatically.

### 3. ObjectSpec has grown to 25+ fields, 666 lines

**Where**: `ObjectSpec.scala:60-82, 127-617`
**Impact**: Low — the case class is internally cohesive (all CLI-parsed parameters). But at 25
fields, adding new parameters requires touching the case class, the for-comprehension in parse(),
and the constructor call — three places where a miss silently drops the field. Already happened:
`preset`/`angle`/`seed` were in ValidKeys for Sprin 30 but never stored.
**Effort**: 4h

**What**: The case class has grown from ~12 fields in Sprint 20 to 25+ fields. The parse()
method for-comprehension is ~20 lines with ~15 parse functions. Each new CLI key adds a field,
a parse function, and a constructor line — three points of coupling.

**Suggested direction**: Group related fields into sub-case-classes (LSystemParams,
ConeParams, PlaneParams). The parse() method delegates to sub-parsers, each returning its
own case class. Reduces the coupling surface and makes "forgot to wire the field" errors
compile-time failures.

## Resolved since last review

| ID | Resolution |
|----|-----------|
| L-upload-texture-file-raw-int | ✅ Now throws `TextureUploadException` (Sprint 31.6) |
| L-project4d-async-error | ✅ Async error contract documented in file header (Sprint 31.6) |
| AI: ser_enabled bool→uint32_t | ✅ Fixed (Sprint 31.8) |
| AI: last_*_count reset on re-init | ✅ Fixed (Sprint 31.8) |
| AI: dead isSerSupported removed | ✅ Removed (Sprint 31.8) |
| Denoise-per-frame (finding #1) | ✅ Fixed — configureOutputMode called in render() paths |
| CLI accumulation flag (finding #2) | ✅ --accumulation-frames CLI flag added |
| Curve ignores fields (finding #3) | ✅ Texture/rotation now applied when supported |

---

# Code Quality Improvements — Open Issues

**Last Updated:** 2026-06-29

Resolved items are removed from this file entirely — git history is the record of what was fixed.

---

## High Priority

*(none)*

---

## Medium Priority

<<<<<<< HEAD
| ID | Issue | Location | Effort |
|----|-------|----------|--------|
| M-lsystem-duplication | LSystemTurtle3D/4D share ~80% duplicated algorithm. Extract shared base trait parameterized by point type. | `LSystemTurtle3D.scala`, `LSystemTurtle4D.scala` | 4h |
| M-objectspec-bloat | ObjectSpec at 25+ fields, 666 lines. Group related fields into sub-case-classes. | `ObjectSpec.scala:60-82` | 4h |
=======
*(all resolved — Sprint 30)*
>>>>>>> origin/main

---

## Low Priority

<<<<<<< HEAD
| ID | Issue | Location | Effort |
|----|-------|----------|--------|
| L-turtle4d-emitrun | LSystemTurtle4D.emitRun has no minimum-points guard (3D turtle has ≥2). Add check and unit test. | `LSystemTurtle4D.scala:211` | 15min |
=======
| ID | Issue | Location |
|----|-------|----------|
| L-upload-texture-file-raw-int | `uploadTextureFromFile` returns raw negative `Int` on failure while `uploadTexture` throws `TextureUploadException`. Behavior change needed: three production callers treat negative index as "skip and continue". Deferred until fail-fast vs graceful-skip decision is made. | `optix-jni/.../OptiXTextureApi.scala:67` |
| L-project4d-async-error | `cudaGetLastError()` at `project4d.cu:147` only captures launch-configuration errors, not async kernel execution errors. The required `cudaDeviceSynchronize` is in the caller, not the callee — document the contract or move the sync inside. | `menger-geometry/src/main/native/project4d.cu:147` |

---

## Tooling Gaps

*(all gaps identified in prior reviews have been closed)*

| Tool | Status | Where |
|------|--------|--------|
| ArchUnit | Closed — 14+6 active rules | `ArchitectureSpec.scala`, `ArchitecturePhase2Spec.scala` |
| cppcheck | Closed — pre-push + CI | `.cppcheck-suppress`, `.git_hooks/pre-push`, `.gitlab-ci.yml` |
| clang-tidy | Closed — pre-push + CI | `.clang-tidy`, `.git_hooks/pre-push`, `.gitlab-ci.yml` |

>>>>>>> origin/main
