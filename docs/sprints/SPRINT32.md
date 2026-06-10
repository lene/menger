# Sprint 32: Spectral Dispersion

**Sprint:** 32 - Spectral Dispersion
**Status:** Not Started
**Estimate:** ~22 hours
**Branch:** `feature/sprint-32`
**Dependencies:** None hard. Denoiser (Sprint 29) already available — dispersion adds
per-wavelength noise that accumulation + denoising absorb.
**Feature ID:** F4 in [FEATURE_DEPENDENCIES.md](FEATURE_DEPENDENCIES.md)

---

## Goal

Wavelength-dependent refraction: white light splits into spectral colors through
prisms, diamonds, and glass sponges ("fire"). Implemented via hero-wavelength sampling
with a Cauchy IOR model — zero cost and zero image change for non-dispersive materials.

---

## Success Criteria

- [ ] A glass prism in white directional light shows a rainbow spread in refracted
      output
- [ ] `material=diamond` with dispersion enabled shows colored fire absent in current
      renders
- [ ] `dispersion = 0` (default) renders bit-identical to current output — all
      existing reference images unchanged
- [ ] Spectral noise converges under existing `accumulation`; documented sample
      recommendations
- [ ] All tests pass

---

## Tasks

### Task 32.1: Dispersion Material Parameter + IOR Model

**Estimate:** 4h

**Implementation:**
- Material parameter: Abbe number `V_d` (the standard optics spec — lower = more
  dispersive). `dispersion = 0` (sentinel) means "off", keeping the default path
  untouched
- Cauchy model in the shader: `n(λ) = A + B/λ²` with `A`, `B` derived once CPU-side
  from the material's `ior` (defined at the d-line, 587.6 nm) and `V_d`:
  `B = (n_d − 1) / (V_d · (1/λ_F² − 1/λ_C²))`, `A = n_d − B/λ_d²`
  (λ_F = 486.1 nm, λ_C = 656.3 nm). Precompute A and B into `InstanceMaterial`;
  the shader evaluates one multiply-add per refraction
- Preset values: crown glass V_d ≈ 59, flint glass V_d ≈ 36, water V_d ≈ 56,
  diamond V_d ≈ 33 (high dispersion — the source of fire). Presets keep
  `dispersion = 0` by default; new preset variants `glass-dispersive`,
  `diamond-dispersive` opt in
- CLI: `--objects 'type=sphere:material=diamond:dispersion=33'`; DSL:
  `Material(ior = 2.42f, dispersion = 33f)`

---

### Task 32.2: Hero-Wavelength Sampling in Shaders

**Estimate:** 8h

**Implementation:**
- Per camera-ray sample, draw one "hero" wavelength λ ∈ [380, 730] nm, stratified
  across accumulation frames (frame index drives the stratum → deterministic under
  fixed seed, converges fast)
- The wavelength travels in the per-ray payload (one extra float). Rays that never
  hit a dispersive material ignore it — their path is wavelength-independent and the
  full RGB result is computed as today
- On the **first** refraction at a dispersive surface, the path becomes spectral:
  refraction direction uses `n(λ)`, and the path's RGB throughput is multiplied by
  the CIE response for λ (analytic XYZ fitting functions — Wyman et al. 2013 —
  evaluated in-shader, then XYZ→sRGB matrix) times a normalization factor of 3 (one
  wavelength carries the energy of the full spectrum estimator)
- Subsequent refractions on the same path reuse the same λ (correct hero-wavelength
  behavior; no chromatic decorrelation inside one path)
- Reflection components (Fresnel) stay RGB — only the refracted branch is spectral.
  This is an approximation that keeps metallic/reflective code untouched and is
  visually correct for the target effects (prisms, fire, rainbow caustics later)

**Risk note:** payload size increase — check against the 8-payload-register budget
in the existing pipeline before coding; if tight, pack λ as 16-bit normalized into an
existing spare slot.

---

### Task 32.3: Accumulation + Denoiser Interplay

**Estimate:** 3h

- With `dispersion > 0`, a single frame is visibly color-noisy; document recommended
  `accumulation ≥ 16` for stills
- Verify the Sprint 29 denoiser handles spectral noise (it does for HDR noise in
  general, but confirm no hue smearing on the prism reference scene; record findings)
- `--stats` gains a spectral-rays counter for diagnostics

---

### Task 32.4: Presets, Demo Scenes

**Estimate:** 2h

- `examples.dsl.PrismDispersion`: triangular prism (existing mesh path), narrow white
  directional light, dark background
- `examples.dsl.DiamondFire`: dispersive diamond on checkered plane under area light

---

### Task 32.5: Tests + Reference Images + Documentation

**Estimate:** 5h

- Unit: Cauchy coefficient derivation (known glass values), CIE XYZ fit sanity
  (λ=550 nm → green-dominant), λ stratification coverage
- Regression: standard scene set with `dispersion=0` must match existing references
  exactly (this is the critical guarantee)
- Integration: prism reference image (accumulation 32, fixed seed),
  dispersive-diamond reference
- `scripts/manual-test.sh`: prism + diamond fire (append at end)
- User guide: Spectral Dispersion section (Abbe number table, noise guidance);
  CHANGELOG.md entry

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 32.1 | Dispersion parameter + Cauchy IOR model | 4h |
| 32.2 | Hero-wavelength sampling in shaders | 8h |
| 32.3 | Accumulation + denoiser interplay | 3h |
| 32.4 | Presets + demo scenes | 2h |
| 32.5 | Tests + reference images + documentation | 5h |
| **Total** | | **~22h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green — including unchanged references for dispersion=0
- [ ] CHANGELOG.md updated
- [ ] Integration + manual test scripts cover dispersion scenes
