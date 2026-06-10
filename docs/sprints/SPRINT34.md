# Sprint 34: Production-Quality Caustics

**Sprint:** 34 - Production-Quality Caustics
**Status:** Not Started
**Estimate:** ~24 hours
**Branch:** `feature/sprint-34`
**Dependencies:** Sprint 32 (spectral machinery enables dispersive caustics, task 34.4).
The PPM tuning investigation (docs/caustics/, CausticsRenderer.cpp) is the input state.
**Feature ID:** F16 in [FEATURE_DEPENDENCIES.md](FEATURE_DEPENDENCIES.md)

---

## Goal

Turn the long-running progressive-photon-mapping caustics experiment into a shipped,
documented feature: tuned defaults locked by the existing reference ladder, automatic
parameter derivation from scene properties, finalized CLI/DSL surface, and — building
on Sprint 32 — dispersive (rainbow) caustics.

---

## Success Criteria

- [ ] Caustics on the standard glass-sphere reference scene match the path-traced
      reference within the tolerance defined by the test ladder
      (docs/caustics/CAUSTICS_TEST_LADDER.md), with **default** parameters
- [ ] `--caustics` with no further parameters produces good results on all ladder
      scenes (auto-tuning)
- [ ] Dispersive caustics: white light through a dispersive glass sphere produces a
      rainbow-fringed caustic
- [ ] Caustics off by default; existing references unchanged
- [ ] All tests pass

---

## Tasks

### Task 34.1: Conclude the PPM Parameter Investigation

**Estimate:** 6h

**Implementation:**
- Resume from the current investigation state (docs/caustics/CAUSTICS_ITERATION_LOG.md
  and the active parameter notes); run the remaining parameter grid on the reference
  ladder scenes
- Lock defaults: photon count, initial gather radius, radius-shrink alpha
  (PPM α, typically 0.7), iteration count coupling to `accumulation`
- Record the decision + measured ladder results in CAUSTICS.md; close
  CAUSTICS_FIX_PLAN.md items that this resolves; delete superseded analysis docs
  (per repo policy: resolved findings are deleted, not archived)

---

### Task 34.2: Auto-Tuning Heuristics

**Estimate:** 5h

Derive parameters from the scene instead of requiring hand-tuning:

- Initial gather radius from scene scale: `r₀ = k · bboxDiagonal` with `k` calibrated
  on the ladder (separately for plane-scale and object-scale caustics)
- Photon budget from light count and the solid angle subtended by transparent
  geometry from each light (cheap estimate: bounding-sphere of refractive instances
  seen from light position) — lights that can't see glass get no photon budget
- Iterations: tie to `RenderSettings.accumulation` (one PPM iteration per
  accumulation frame — radii shrink across frames, matching PPM's convergence model)
- Every heuristic is overridable by the explicit parameters in 34.3; `--stats`
  reports the derived values so users can inspect what auto-tuning chose

---

### Task 34.3: CLI/DSL Surface Finalization

**Estimate:** 4h

**Implementation:**
- CLI: `--caustics` (auto-tuned), `--caustics photons=2000000:radius=0.05` for
  explicit control
- DSL: finalize `Caustics(photons: Option[Int], radius: Option[Float],
  alpha: Float = 0.7f)` in `menger.dsl.Caustics` — `None` = auto (34.2)
- Validation: warn (don't fail) when caustics are enabled with no transparent
  objects or no shadow-capable lights; document interplay with
  `--transparent-shadows`
- Make sure the existing `examples.dsl.CausticsDemo` / `CausticsReference*` scenes
  use the finalized API

---

### Task 34.4: Dispersive Caustics

**Estimate:** 5h

**Implementation:**
- Photon wavelengths: sample each photon's λ with the Sprint 32 stratification;
  photon refraction through dispersive materials uses `n(λ)`; photon carries RGB
  energy = CIE response for λ (same conversion as camera-side hero wavelengths)
- Non-dispersive scenes: λ never alters refraction, RGB energy sums to white —
  bit-compatible with 34.1 results (verify on the ladder)
- Reference scene: white directional light through a dispersive glass sphere onto a
  neutral plane — rainbow-fringed ring
- Photon-count guidance: spectral photons need ~4× budget for equal smoothness;
  auto-tuning multiplies the budget when the scene contains dispersive materials

---

### Task 34.5: Reference Ladder → Integration Suite + Documentation

**Estimate:** 4h

- Promote the CAUSTICS_TEST_LADDER scenes into `scripts/integration-tests.sh` as
  gated scenarios (tolerances from 34.1's locked results)
- `scripts/manual-test.sh`: caustic + dispersive-caustic scenes (append at end)
- User guide: Caustics section (when PPM beats the default path, auto vs. explicit
  parameters, dispersive example); consolidate docs/caustics/ to CAUSTICS.md +
  CAUSTICS_REFERENCES.md, delete the rest if resolved
- CHANGELOG.md entry; arc42 §9 decision record (PPM parameters + auto-tuning)

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 34.1 | Conclude PPM investigation, lock defaults | 6h |
| 34.2 | Auto-tuning heuristics | 5h |
| 34.3 | CLI/DSL surface finalization | 4h |
| 34.4 | Dispersive caustics | 5h |
| 34.5 | Reference ladder → integration suite + docs | 4h |
| **Total** | | **~24h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] CHANGELOG.md updated
- [ ] Caustics ladder scenarios gated in the integration suite
