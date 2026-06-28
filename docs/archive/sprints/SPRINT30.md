# Sprint 30: OptiX API Coverage II — Architecture Hardening, 1.0 Prep

**Sprint:** 30 - OptiX API Coverage II
**Status:** 🔄 In Progress
**Estimate:** ~55 hours
**Branch:** `feature/sprint-30`
**Dependencies:** Sprint 29 (denoiser + curves establish the API-expansion pattern)
**Feature ID:** F13 Phase 2 in [FEATURE_DEPENDENCIES.md](FEATURE_DEPENDENCIES.md)

---

## Goal

Architecture hardening sprint: thorough architectural review with concrete follow-through on
critical findings, optix-jni 1.0 readiness, and resolution of structural code-quality issues
identified in both the Sprint 29 code review and the Sprint 30 architectural review. Motion blur
deferred to Sprint 31+ to make room for architectural debt resolution.

---

## Success Criteria

- [ ] Architectural review completed and findings addressed (ARCHITECTURE_REVIEW.md)
- [ ] Critical arch-review findings resolved: curve in VALID_TYPES, native guards, per-frame buffer fix
- [ ] InteractiveEngine god-object refactored: RotationFastPath extracted, dispatch unified
- [ ] Performance governance installed: real baselines measured, PerfCheck blocking, GPU tunables
- [ ] arc42 all 12 sections reviewed for accuracy; duplicate AD numbers fixed; stale claims corrected
- [ ] Audit document lists every OptiX 8.x/9.x feature group with expose/defer/rationale
- [ ] Validation mode + SER implemented
- [ ] optix-jni 1.0 readiness checklist complete (API review, Scaladoc, MiMa baseline)
- [ ] All tests pass

---

## Execution Order (dependency-aware)

```
Phase A — Critical fixes (minutes-hours):
  A1. Fix "curve" ∉ VALID_TYPES
  A2. Fix M-native-no-guard (isInitialized checks)
  A3. Fix per-frame buffer re-alloc (guard on size change)

Phase B — Implementation (hours):
  30.4  Validation mode + SER          ← benefits from A2 native guards
  30.8  InteractiveEngine refactor     ← benefits from A1 dispatch fix
  30.9  Performance governance         ← independent
  30.2  arc42 coherence pass           ← benefits from A1-A3 + arch review

Phase C — Polish (hours):
  30.5  optix-jni 1.0 readiness        ← benefits from API audit doc
  30.7  CODE_IMPROVEMENTS remaining     ← 4 findings instead of 6 (A1-A3 knock out 2)
  30.6  Tests + documentation          ← depends on implementation
```

---

## Tasks

### Task A1–A3: Critical Arch-Review Fixes

**Estimate:** 2h

Three fixes from the architecture review that block or enable subsequent tasks:

- **A1 — `"curve"` missing from `ObjectType.VALID_TYPES`** (15min): Add `"curve"` to
  `menger-common/.../ObjectType.scala` VALID_TYPES. The DSL path works but the CLI `type=curve`
  path silently rejects it. Critical: a Sprint 29 feature is broken for CLI users.
- **A2 — Native guard for `setDenoisingEnabled`/`setAccumulationFrames`** (1h): Add Scala wrapper
  methods with `require(isInitialized)` precondition checks. Currently `public native` with no
  guard — calls before init or after dispose cause native SIGSEGV. (CODE_IMPROVEMENTS
  `M-native-no-guard`)
- **A3 — Per-frame buffer re-allocation in render hot path** (1h): Guard `cudaFree` +
  `cudaMalloc` + `cudaMemcpy` on actual size changes in `OptiXWrapper::render()` IAS path.
  Eight geometry-data arrays are freed/re-allocated unconditionally every frame despite a
  design-intent comment to only do so on size change. (Arch review finding #5)

---

### Task 30.1: Architectural Review ✅

**Estimate:** 6h — **DONE**

Full architectural review using the 4-axis framework. Output: `ARCHITECTURE_REVIEW.md` (2026-06-27).
8 findings across soundness/maturity/evolvability/performance. Three findings promoted to A1-A3;
two promoted to 30.8 and 30.9.

---

### Task 30.2: arc42 Review and Coherence Pass

**Estimate:** 3h

Systematic read-through of all 12 arc42 sections against current codebase. Key items:

- Fix duplicate AD-24/AD-25 numbers (Sprint 29 decisions need AD-27, AD-28)
- §10: Add "❌ Not Implemented" markers to each caustics C1-C8 rung
- §10: Mark perf budgets as aspirational until measured baselines (30.9)
- §5: Distinguish enforced from aspirational dependency claims
- §11: Update JNI leak risk status to match reality (gate is stubbed)
- Sprint 28-29 decisions present? All sections current?
- Update "Last Updated" dates

---

### Task 30.3: OptiX API Audit ✅

**Estimate:** 4h — **DONE**

Output: `docs/optix-api-audit.md`. 22 OptiX 9.0 feature groups catalogued (11 exposed, 2 planned,
9 deferred with rationale). 1.0 scope boundary defined. Also serves as arc42 §9 decision record
candidate.

---

### Task 30.4: Validation Mode + SER

**Estimate:** 6h

- **Validation mode:** `OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL` toggle via
  `MENGER_OPTIX_VALIDATION=1` env var / library API flag. Catches SBT and payload errors at
  call site. Document in TROUBLESHOOTING.md.
- **Shader execution reordering:** `optixReorder` call in raygen on RTX 40xx+ (Ada), guarded by
  device-capability check. Benchmark with Sprint 28 set; if <5% gains, expose API but leave off
  by default.

---

### Task 30.5: optix-jni 1.0 Readiness

**Estimate:** 5h

- Public API review: every public trait/class/method → keep/rename/deprecate
- Scaladoc completeness CI gate (fail on undocumented public API)
- MiMa baseline against 0.1.2 release; block future releases on MiMa failures
- Remove all deprecated API before 1.0
- Output: 1.0 release checklist in optix-jni repo

---

### Task 30.6: Tests + Documentation

**Estimate:** 3h

- Validation-mode smoke test (assert clean launch)
- CLI `type=curve` integration test (exercises the A1 fix)
- `scripts/manual-test.sh`: SER entry if gains measurable
- User guide: Validation Mode section; curve CLI documentation
- TROUBLESHOOTING.md update
- CHANGELOG.md entry

---

### Task 30.7: CODE_IMPROVEMENTS Resolution

**Estimate:** 8h (reduced from 10h — A1+A2 resolve 2 findings)

**Medium (remaining):**
- **M-per-frame-denoise** (2h): Call `configureOutputMode` per-frame in `WithAnimation.render()`,
  `WithPreview.render()`, `CliAnimationEngine.render()`.
- **M-cli-accumulation-flag** (2h): Add `--accumulation-frames <int>` CLI option to
  `MengerCLIOptions`, thread through `Main.scala` `--object` path.
- **M-curve-ignores-fields** (3h): Guard against or implement texture/rotation/normalMap in
  `CurveSceneBuilder`. Silently ignores DSL fields.

**Low:**
- **L-curve-error-untested** (1h): Scalamock test for `CurveSceneBuilder.buildScene` error path.

---

### Task 30.8: InteractiveEngine God-Object Refactor

**Estimate:** 9h — **NEW** (from arch review finding #6)

**Goal:** Decompose the 662-line `InteractiveEngine` mixing 8 concerns.

- **30.8a — Extract `RotationFastPath`** (3h): Move 5 copy-paste 4D fast-path methods
  (`tryRotation4DFastPath`, `tryMenger4DFastPath`, etc.) into a strategy object parameterized
  by type. Each variant differs only in `Scene4DCache` case + `renderer.update*Projection` call.
- **30.8b — Single source of truth for dispatch** (3h): Replace dual if-else chains in
  `GeometryRegistry.builderFor` and `InteractiveEngine.buildScene4DTrackedOrFallback` with a
  `Map[ObjectType → Builder]` consumed by both. Fix the divergence where Sierpinski4D and
  Hexadecachoron4D are wired in InteractiveEngine but absent from GeometryRegistry.
- **30.8c — Add ArchUnit cohesion rule** (1h): `noClasses().that().resideInAPackage("menger.engines")`
  `.should().haveSimpleNameContaining("Engine").and().should().haveCodeUnitSize(<= 300)`.
  Ensures engines stay orchestrators, not monoliths.
- **30.8d — Test coverage** (2h): Verify refactored dispatch covers all `VALID_TYPES`;
  verify fast-path extraction preserves 4D animation behavior.

---

### Task 30.9: Performance Governance

**Estimate:** 6h — **NEW** (from arch review finding #4)

**Goal:** Transform performance from aspirational to governed — real baselines, blocking gate,
and runtime GPU tunability.

- **30.9a — Measure real baselines** (2h): Run `benchmark.sh` on the CI GPU runner for all 4
  scenes in `perf-baseline.json`. Replace placeholder `5000.0` ms values. Add sponge-L3 and
  menger4d-L3 entries matching arc42 §10.4 P1/P2 scenes.
- **30.9b — Promote PerfCheck to blocking** (1h): Change `allow_failure: true` → `false` in
  `.gitlab-ci.yml` after 3 stable green runs. Set threshold to 25% regression (not 15% — first
  real baseline needs headroom).
- **30.9c — Runtime GPU tunables** (2h): Add `MENGER_OPTIX_BLOCK_SIZE` and
  `MENGER_OPTIX_STREAMS` environment variables read at `OptiXContext::launch()` time. Default to
  current behavior when unset. Document in TROUBLESHOOTING.md.
- **30.9d — Update arc42 §10.4** (1h): Replace aspirational language with measured baselines.
  Link to `perf-baseline.json`. Mark P1/P2 as "Enforced by PerfCheck CI job."

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| A1–A3 | Critical arch-review fixes | 2h |
| 30.1 | Architectural review (4-axis) | 6h ✅ |
| 30.2 | arc42 coherence pass | 3h |
| 30.3 | OptiX API audit + 1.0 scope | 4h ✅ |
| 30.4 | Validation mode + SER | 6h |
| 30.5 | optix-jni 1.0 readiness | 5h |
| 30.6 | Tests + documentation | 3h |
| 30.7 | CODE_IMPROVEMENTS resolution | 8h |
| 30.8 | InteractiveEngine refactor | 9h |
| 30.9 | Performance governance | 6h |
| **Total** | | **~52h** (+ A1-A3) |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] CHANGELOG.md updated
- [ ] arc42 §9 decision record for the API audit
- [ ] CODE_IMPROVEMENTS.md findings resolved or captured as backlog
- [ ] New ARCHITECTURE_REVIEW.md findings resolved or captured as backlog
