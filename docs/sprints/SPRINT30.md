# Sprint 30: OptiX API Coverage II — Architecture, API Audit, 1.0 Prep

**Sprint:** 30 - OptiX API Coverage II
**Status:** 📋 Planned
**Estimate:** ~27 hours
**Branch:** `feature/sprint-30`
**Dependencies:** Sprint 29 (denoiser + curves establish the API-expansion pattern)
**Feature ID:** F13 Phase 2 in [FEATURE_DEPENDENCIES.md](FEATURE_DEPENDENCIES.md)

---

## Goal

Architecture hardening sprint: thorough architectural review across all 4 axes with
concrete follow-through on findings, systematic arc42 audit and coherence pass,
OptiX API surface audit with expose/defer decisions, and the concrete readiness
plan for optix-jni 1.0 (SemVer stability contract). Motion blur deferred to
Sprint 31 (or later) to make room for architectural debt resolution.

---

## Success Criteria

- [ ] Architectural review completed against the 4-axis framework (Soundness,
      Maturity, Evolvability, Performance Architecture); report in ARCHITECTURE_REVIEW.md
- [ ] arc42 all 12 sections reviewed for accuracy and completeness; drift from code
      identified and corrected; stale claims replaced with current state
- [ ] Audit document lists every OptiX 8.x/9.x feature group with status
      exposed / planned / deliberately-not-exposed + rationale
- [ ] optix-jni 1.0 readiness checklist complete (API review, Scaladoc, MiMa baseline)
- [ ] All tests pass

---

## Tasks

### Task 30.1: Architectural Review

**Estimate:** 6h

Full architectural review using the 4-axis framework (Soundness, Maturity,
Evolvability, Performance Architecture) from `arch-review` skill.
Output: `ARCHITECTURE_REVIEW.md` at repo root.

Scope: whole-system structure across menger-app, menger-geometry, optix-jni,
and menger-common. Anchored to arc42 §5, §9, §10, §11. Each finding must
include a fitness-function verdict: is the invariant guarded, partially
guarded, or unguarded by an automated check?

Top-priority follow-through on findings that can be resolved within the sprint
(fitness functions, docs, minor refactors). Larger findings become backlog items.

---

### Task 30.2: arc42 Review and Coherence Pass

**Estimate:** 3h

Systematic read-through of all 12 arc42 sections against current codebase:

- §1 Introduction: stakeholder interests still accurate?
- §2 Constraints: TC-* still match reality?
- §3 Context: external system boundaries unchanged?
- §4 Solution Strategy: still the approach?
- §5 Building Block View: module graph, interfaces, dependencies current?
- §6 Runtime View: key scenarios (startup, render, animation, publish)?
- §7 Deployment: Docker, CI, cloud still accurate?
- §8 Cross-cutting Concepts: all conventions documented?
- §9 Architectural Decisions: new ADs from Sprint 28-29 present?
- §10 Quality: fitness functions current? Caustics ladder still 0/8?
- §11 Risks: new risks from arch review captured?
- §12 Glossary: terms up to date?

Fix errors and stale claims. Flag sections needing major rework as backlog items.
Update "Last Updated" dates. Goal: arc42 is once again a trustworthy source.

---

### Task 30.3: OptiX API Audit

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

### Task 30.4: Validation Mode + SER

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

### Task 30.5: optix-jni 1.0 Readiness

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

### Task 30.6: Tests + Documentation

**Estimate:** 3h

- Validation-mode smoke test (assert clean run)
- `scripts/manual-test.sh`: SER on/off entry if gains measurable
- User guide: Validation Mode section; update TROUBLESHOOTING.md
- CHANGELOG.md entry

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 30.1 | Architectural review (4-axis) | 6h |
| 30.2 | arc42 coherence pass (all 12 sections) | 3h |
| 30.3 | OptiX API audit + 1.0 scope definition | 4h |
| 30.4 | Validation mode + SER | 6h |
| 30.5 | optix-jni 1.0 readiness | 5h |
| 30.6 | Tests + documentation | 3h |
| **Total** | | **~27h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] CHANGELOG.md updated
- [ ] arc42 §9 decision record for the API audit
- [ ] CODE_IMPROVEMENTS.md findings from arch review captured
