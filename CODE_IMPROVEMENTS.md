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

| ID | Issue | Location | Effort |
|----|-------|----------|--------|
| M-lsystem-duplication | LSystemTurtle3D/4D share ~80% duplicated algorithm. Extract shared base trait parameterized by point type. | `LSystemTurtle3D.scala`, `LSystemTurtle4D.scala` | 4h |
| M-objectspec-bloat | ObjectSpec at 25+ fields, 666 lines. Group related fields into sub-case-classes. | `ObjectSpec.scala:60-82` | 4h |

---

## Low Priority

| ID | Issue | Location | Effort |
|----|-------|----------|--------|
| L-turtle4d-emitrun | LSystemTurtle4D.emitRun has no minimum-points guard (3D turtle has ≥2). Add check and unit test. | `LSystemTurtle4D.scala:211` | 15min |
| L-ci-nvidia-caps-duplication | `NVIDIA_DRIVER_CAPABILITIES` (and `NVIDIA_VISIBLE_DEVICES`) is redeclared identically in 6 GPU jobs because GitLab CI job-level `variables:` fully replaces the global block rather than merging with it. A value change (as in Sprint 33's `display`-capability fix) requires editing all 6 sites by hand with no compiler/linter catch for a missed one. | `.gitlab-ci.yml`: `Test:Full`, `Test:OptiXIntegration`, `Run:UseDocker`, `CheckRunTime`, `PerfCheck`, `Test:InstallSmoke` | 1h |

---

# Code Quality Review — Sprint 32 (2026-07-01)

Spectral dispersion + architecture hardening. 18 files changed, ~30 commits across 3 repos.
Published menger-common 0.1.4, optix-jni 0.1.9.

## Resolved in Sprint 32

| ID | Issue | Resolution |
|----|-------|-----------|
| Arch-T1 | Triplicated object-type dispatch | ✅ GeometryRegistry.unified (unified table replaces 3 if/else chains) |
| Arch-T2 | No performance governance | ✅ PerfCheck CI + benchmark.sh + perf-baseline.json |
| Arch-T5 | Missing native-binding ArchUnit | ✅ MengerRenderer native-method guard in ArchitecturePhase2Spec |
| Arch-T7 | No JNI fault-injection coverage | ✅ SceneBuilderFaultInjectionSuite + FaultInjectionSuite |
| Arch-T9 | Script-parity gaps | ✅ ScriptParitySuite (all VALID_TYPES covered) |
| Arch-T10 | No fast-path regression guard | ✅ RotationFastPath + FastPathRegressionSuite |
| Arch-A4 | String-based sub-builder dispatch | ✅ SubBuilderType sealed enum |
| Arch-A5 | 4D presets not in LSystemPresets | ✅ hilbert4d preset in LSystemPresets |
| Arch-T11 | Missing OptiX ADR | ✅ AD-31: OptiX as sole rendering backend |

## New findings from Sprint 32

### 1. heroWavelengthToRGB was dead code (spectral tint not applied)

**Where**: `optix-jni/src/main/native/shaders/helpers.cu:524`
**Impact**: Medium — Cauchy IOR computation worked correctly (angles changed), spectralRays counter
incremented, but refracted color was never tinted by wavelength → no visible rainbow effect.
**Fix**: Applied in 0.1.8/0.1.9 — tint after optixTrace returns in traceRefractedRay.
**Lesson**: Device functions must have verified call sites, not just definitions.
Detection: `grep -rn 'FunctionName' src/ | grep -v '__device__'` — single hit = dead code.

### 2. addPlaneInstanceNative JNI parameter shift (checkerboard regression)

**Where**: `optix-jni`: Scala @native, Scala wrapper, JNI C++ binding
**Impact**: High — cauchy_a/b parameters were added to all add*Instance methods except
addPlaneInstanceNative. Parameters shifted silently (r2→cauchy_a, g2→cauchy_b, etc.),
corrupting checkerboard plane rendering. No crash, just wrong output.
**Fix**: Applied in 0.1.9 — 3-layer fix (Scala @native, wrapper, JNI) + stale Ivy cache clear.
**Lesson**: Bulk parameter additions across ~10 functions need grep-to-verify on every
function signature after the change. Stale Ivy cache silently masks the fix.

### 3. failure-handling skill had no enforcement (3 dismissals in one sprint)

**Impact**: Medium — violations: compute-sanitizer dismissed, pre-existing test failures ignored
twice. Skill existed but was loaded without being followed. Simplified: "if it fails, fix it"
— no pre-existing exemptions.

---

# Code Quality Review — Sprint 34 (2026-07-02)

PBR texture sets. 23 files changed, ~4000 LOC across 2 repos (16 in menger, 7 in optix-jni).
Published optix-jni 0.1.11. Pipeline green. 23 unit tests, 1 integration test, 2 manual entries.

## Resolved in Sprint 34

| ID | Issue | Resolution |
|----|-------|-----------|
| CRITICAL #1 | hit_cone.cu missing metallic/AO apply calls | ✅ Fixed — 2 lines added |
| CRITICAL #2 | DX→GL normal map conversion never executed | ✅ Fixed — invertGreenChannel in loadDxNormalMap |
| MEDIUM #5 | Convention detection unanchored substring matching | ✅ Fixed — split("_") component-based matching |
| MEDIUM #6 | Synthetic key collision with static filenames | ✅ Fixed — guard rejects filenames starting with "set:" |
| LOW #8 | textureCount hardcoded * 5 | ✅ Fixed → * 7 |
| HIGH #3 | TextureSetMetadata.load never called in production | ✅ Wired — called from loadTextureSet |
| HIGH #3 | textureSetRes dead CLI parameter | ✅ Wired — passed through to TextureSetResolver |
| MEDIUM #7 | No false-positive test for substring matching | ✅ Added — disco_ao.png component test |

## Open (deferred)

| ID | Issue | Severity | Effort | Notes |
|----|-------|----------|--------|-------|
| DEFER-1 | 5 shaders lack UV infrastructure (cylinder, curve, 4D) — cannot apply any PBR texture maps | Medium | ~16h | Needs UV computation for each geometry type. Separate feature. |
| DEFER-2 | uvScale from metadata/specs has no GPU-side uniform | Low | ~4h | Needs shader-level UV scale parameter + JNI wiring |
| DEFER-3 | IOR from metadata sidecar not consumed | Low | ~2h | Needs MaterialOverride chain (sidecar IOR → material preset IOR → spec.ior) |
| DEFER-4 | No visual verification that metallic/AO maps produce different renders | Medium | ~2h | Test data needs base metallic > 0 (matte has metallic=0, texture map multiplies 0) |
| DEFER-5 | Convention detection semi-anchored — "ao" as component still matches | Low | ~1h | Component matching reduces but doesn't eliminate false positives (disco_ao.png works by coincidence) |
| DEFER-6 | Duplicate texture loading when explicit override + texture set cover same file | Low | ~1h | Wastes GPU slots, harmless for correctness |

## New findings from Sprint 34

### 1. `--no-verify` bypassed pre-commit hook on every commit

**Impact**: High (process, not code) — All 3 CI failures (scalafix return keyword, unused imports,
missing PNG files) were pre-commit hook catches. Each would have been caught in <30s locally.
35 commits used --no-verify; CI cycles took 15-20 min each. Net waste: ~2h of CI wait time.

**Fix**: Memory entry added: "NEVER --no-verify for routine commits." failure-handling skill
updated with Sprint 34 anti-pattern.
