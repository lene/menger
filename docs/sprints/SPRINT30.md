# Sprint 30: OptiX API Coverage II — Motion Blur, API Audit, 1.0 Prep

**Sprint:** 30 - OptiX API Coverage II
**Status:** Not Started
**Estimate:** ~26 hours
**Branch:** `feature/sprint-30`
**Dependencies:** Sprint 29 (denoiser + curves establish the API-expansion pattern)
**Feature ID:** F13 Phase 2 in [FEATURE_DEPENDENCIES.md](FEATURE_DEPENDENCIES.md)

---

## Goal

Second phase of OptiX API coverage: a systematic audit of the remaining OptiX API
surface with an expose/defer decision per feature, transform motion blur (the most
valuable audited item — it upgrades every animation render), and the concrete
readiness plan for optix-jni 1.0 (SemVer stability contract).

---

## Success Criteria

- [ ] Audit document lists every OptiX 8.x/9.x feature group with status
      exposed / planned / deliberately-not-exposed + rationale
- [ ] `MotionBlur(shutter = 0.5)` in an animated DSL scene produces visible,
      physically plausible motion blur on rotating geometry
- [ ] Motion blur off by default; reference images unchanged
- [ ] optix-jni 1.0 readiness checklist complete (API review, Scaladoc, MiMa baseline)
- [ ] All tests pass

---

## Tasks

### Task 30.1: OptiX API Audit

**Estimate:** 4h

Enumerate the full OptiX API surface against what optix-jni exposes.

**Feature groups to audit:** acceleration structures (compaction, updates, motion),
curves variants (linear/quadratic/Catmull-Rom/ribbons), sphere primitive, instancing
features, denoiser variants (temporal, AOV, upscaling), opacity micromaps, displaced
micro-meshes, shader execution reordering (SER), payload semantics, validation mode,
multi-GPU, demand-loaded textures.

**Output:** `optix-jni` doc page (and arc42 §9 decision record) with a table:
feature → status → rationale → target sprint if planned. This is the authoritative
1.0 scope definition: 1.0 does **not** mean "everything wrapped", it means "everything
we chose to expose is stable".

---

### Task 30.2: Transform Motion Blur

**Estimate:** 8h

**Implementation:**
- `OptixMotionOptions` on the IAS (numKeys=2, t0=0, t1=1); per-instance
  `OptixMatrixMotionTransform` with begin/end transforms
- Animation integration: for each output frame at animation time `t`, the engine
  computes transforms at `t` and `t + shutter·Δt` and feeds both keys; rays sample
  `optixTrace` time uniformly per sample — accumulation frames stratify the shutter
  interval, so motion blur quality scales with existing `accumulation` setting
- Scope: transform (rigid-body) motion blur only. Deformation/vertex motion blur
  (e.g. for morphing parametric surfaces) is deferred — audit notes it
- DSL: `RenderSettings(motionBlur: Option[MotionBlur])`, `MotionBlur(shutter: Float)`
  (shutter as fraction of frame interval, 0..1)
- 4D note: 4D-rotation animations currently refit projected vertices per frame
  (in-place projection kernel). Vertex refit + motion transforms don't compose —
  for 4D objects, motion blur falls back to camera/object 3D-transform blur only;
  document the limitation

---

### Task 30.3: Validation Mode + SER

**Estimate:** 6h

Two audited items worth implementing immediately:

- **Validation mode:** `OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL` toggle via
  `MENGER_OPTIX_VALIDATION=1` env var / library API flag — catches SBT and payload
  errors at a precise call site instead of CUDA error 718 at launch. Big
  debugging-experience win for library consumers; document in TROUBLESHOOTING.md
- **Shader execution reordering:** `optixReorder` call in raygen on RTX 40xx+
  (Ada) hardware, guarded by a device-capability check; measure with the Sprint 28
  benchmark set — expect gains on divergent scenes (sponges with mixed materials).
  If gains are <5 % on the benchmark set, expose the API but leave it off by default
  and record the measurement

---

### Task 30.4: optix-jni 1.0 Readiness

**Estimate:** 5h

**Implementation:**
- Public API review: every public trait/class/method in optix-jni gets a
  keep / rename / deprecate decision (`OptiXRenderer`, `NativeOptiXApi`, all public
  traits per the pre-1.0 contract in ROADMAP.md)
- Scaladoc completeness gate in the optix-jni CI (fail on undocumented public API)
- MiMa baseline established against the release from Sprint 29; from 1.0 on, MiMa
  failures block release
- Remove all deprecated API before 1.0
- Output: 1.0 release checklist in the optix-jni repo; the release itself happens
  when the checklist is green, not necessarily inside this sprint

---

### Task 30.5: Tests + Documentation

**Estimate:** 3h

- Integration: motion-blurred rotating sponge reference image (fixed seed +
  accumulation makes it deterministic); validation-mode smoke test (assert clean run)
- `scripts/manual-test.sh`: motion blur on/off pair (append at end)
- User guide: Motion Blur section; TROUBLESHOOTING.md validation-mode section
- CHANGELOG.md entry

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 30.1 | OptiX API audit + 1.0 scope definition | 4h |
| 30.2 | Transform motion blur | 8h |
| 30.3 | Validation mode + SER | 6h |
| 30.4 | optix-jni 1.0 readiness | 5h |
| 30.5 | Tests + documentation | 3h |
| **Total** | | **~26h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] CHANGELOG.md updated
- [ ] arc42 §9 decision record for the API audit
