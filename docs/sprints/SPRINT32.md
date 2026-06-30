# Sprint 32: Spectral Dispersion + Architecture Hardening

**Sprint:** 32 — Spectral Dispersion
**Status:** 📋 Planned
**Estimate:** ~76 hours
**Branch:** `feature/sprint-32`
**Dependencies:** None hard. Denoiser (Sprint 29) already available.
**Feature ID:** F4 in [FEATURE_DEPENDENCIES.md](FEATURE_DEPENDENCIES.md), plus architecture
backlog items T1, T2, T5, T7, T9, T10, T11, A4, A5 from Sprint 30/31 reviews.

---

## Goal

**Primary:** Wavelength-dependent refraction — white light splits into spectral colors
through prisms, diamonds, and glass sponges ("fire"). Hero-wavelength sampling with a
Cauchy IOR model — zero cost and zero image change for non-dispersive materials.

**Secondary:** Close the architecture backlog from Sprint 30/31 reviews — unify dispatch,
add performance governance, harden ArchUnit rules, and add automated quality gates.
Every architecture task converts a convention-maintained invariant into a fitness function.

---

## Success Criteria

### Spectral Dispersion
- [ ] A glass prism in white directional light shows a rainbow spread in refracted output
- [ ] `material=diamond` with dispersion enabled shows colored fire absent in current renders
- [ ] `dispersion = 0` (default) renders bit-identical to current output — all existing
      reference images unchanged
- [ ] Spectral noise converges under existing `accumulation`; documented sample recommendations

### Architecture Hardening
- [ ] sbt 2.0.1 builds and tests pass across all three repos
- [ ] Object-type dispatch uses a single registry table (no more triplicated if/else chains)
- [ ] PerfCheck CI job asserts P1/P2 performance budgets against committed baselines
- [ ] Script-parity test catches `type=<...>` drift between integration and manual test scripts
- [ ] ArchUnit native-binding rule uses module-path scoping; `menger.geometry` renamed
- [ ] Render-determinism test and JNI fault-injection tests pass in CI
- [ ] Fast-path regression guard for 4D animations
- [ ] OptiX-as-sole-backend documented as an architectural decision (ADR)
- [ ] 4D presets unified in LSystemPresets; sub-builder dispatch uses sealed trait
- [ ] All tests pass

---

## Tasks

### Task 32.1: Dependency & Hygiene Updates

**Estimate:** 6h
**Priority:** 🔴 Must go first — sbt affects everything

**32.1a — sbt upgrade attempt (1h):** RESULT — sbt 2.0.0 cannot resolve plugins (`sbt-sonatype`, `sbt-jni` lack `_sbt2_3` artifacts). Staying on sbt 1.12.11. Deferred until plugin ecosystem catches up.

**32.1b — CODE_IMPROVEMENTS.md cleanup (1h):**
- Verify Sprint 31 tasks 31.6 + 31.8 actually resolved the claimed items
- Remove resolved items from CODE_IMPROVEMENTS.md
- Add any new findings discovered during sprint-close

**32.1c — Record OptiX-as-sole-backend ADR — T11 (1h):**
- Add a short note to arc42 §9: the renderer is deliberately hard-wired to OptiX
  across ~9 scene builders; a second backend would require a cross-cutting refactor
- Document the decision, rationale, and when to revisit

---

### Task 32.2: Dispersion Material Parameter + Cauchy IOR Model

**Estimate:** 4h
**Depends on:** 32.1 (build tooling)

**Implementation:**
- Material parameter: Abbe number `V_d` (the standard optics spec — lower = more
  dispersive). `dispersion = 0` (sentinel) means "off", keeping the default path
  untouched
- Cauchy model: `n(λ) = A + B/λ²` with `A`, `B` derived once CPU-side from the
  material's `ior` (defined at the d-line, 587.6 nm) and `V_d`:
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

### Task 32.3: Hero-Wavelength Sampling in Shaders

**Estimate:** 8h
**Depends on:** 32.2 (Cauchy model needed in shader)

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

### Task 32.4: Accumulation + Denoiser Interplay

**Estimate:** 3h
**Depends on:** 32.3 (sampling working)

- With `dispersion > 0`, a single frame is visibly color-noisy; document recommended
  `accumulation ≥ 16` for stills
- Verify the Sprint 29 denoiser handles spectral noise (it does for HDR noise in
  general, but confirm no hue smearing on the prism reference scene; record findings)
- `--stats` gains a spectral-rays counter for diagnostics

---

### Task 32.5: Dispersion Presets + Demo Scenes

**Estimate:** 2h
**Depends on:** 32.3 (renders working)

- `examples.dsl.PrismDispersion`: triangular prism (existing mesh path), narrow white
  directional light, dark background
- `examples.dsl.DiamondFire`: dispersive diamond on checkered plane under area light

---

### Task 32.6: Dispersion Tests + Reference Images + Documentation

**Estimate:** 5h
**Depends on:** 32.3, 32.4, 32.5

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

### Task 32.7: Unify Object-Type Dispatch — T1

**Estimate:** 16h
**Priority:** High
**Depends on:** 32.1 (build tooling)
**Finding:** F1 (from ARCHITECTURE_REVIEW.md 2026-06-12)

**Problem:** `ObjectType` dispatch is triplicated across `GeometryRegistry.builderFor`,
`RenderModeSelector`, and `InteractiveEngine.buildScene4DTrackedOrFallback`. Three
hand-maintained if/else chains that already drifted once (sierpinski4d/hexadecachoron4d
gap fixed in Sprint 27 but root cause remains).

**Implementation:**
- Introduce a single source of truth: a registry `Map` of
  `(predicate, builderFactory)` consumed by all three sites
- `InteractiveEngine` still supplies its recorder callbacks, but the
  *type→builder* decision lives in one place
- Refactor `GeometryRegistry` to own the dispatch table
- Remove the three if/else chains; every new type touches one table + one builder

**Fitness function:** The completeness test from the 2026-06-12 patch (every
`ObjectType.VALID_TYPES` entry resolves to exactly one builder) becomes the
guard over the unified table.

**Done when:** The three if/else chains are gone; adding a type touches only the
unified table and one new builder class.

---

### Task 32.8: Quality Infrastructure

**Estimate:** 17h
**Depends on:** 32.1 (build tooling)

**32.8a — Performance governance — T2 (8h):**
- Wire the existing `scripts/benchmark.sh` + `PerfCheck` CI job to arc42 §10.4
  budgets (P1 <5s, P2 <500ms)
- Commit a `perf-baseline.json` with current measurements
- Add `PerfCheck` as a blocking CI job on MR pipelines
- Emit machine-readable results for trend tracking

**32.8b — Script-parity fitness function — T9 (3h):**
- Create a test that extracts `type=<...>` tokens from `integration-tests.sh`
  and `manual-test.sh`
- Assert `integration ⊇ manual` coverage over `ObjectType.VALID_TYPES`
- Fail CI on divergence (adding a type to only one script)

**32.8c — Determinism + JNI fault-injection — T7 (6h):**
- Render the canonical scene twice, assert byte-identical PNG output
- Force `instanceId = -1` mid-frame failure paths for sphere, mesh, curve,
  cylinder, and cone instance-adding JNI calls
- Verify `Try`→`Failure` propagation with scalamock

---

### Task 32.9: Architecture Hardening

**Estimate:** 10h
**Depends on:** 32.1 (build tooling)

**32.9a — Module-scoped ArchUnit native rule — T5 (6h):**
- Make native-binding ArchUnit rules module-path-based (bindings originate only
  from expected module locations)
- `menger-geometry` classes extend the published optix-jni surface, never duplicate it
- Rename `menger.geometry` → `menger.video` (or assert it has no `@native` methods)
- **Fitness function:** module-path-scoped native-binding ArchUnit rule

**32.9b — Fast-path regression guard — T10 (4h):**
- Add a counter on the renderer tracking projection calls vs instance builds
- Assert an N-frame 4D animation issues O(frames × instances) projection calls
  and O(1) instance builds
- Catch regression to per-frame rebuild (a one-line condition flip that
  could silently degrade performance)

---

### Task 32.10: 4D Hilbert Curve Preset

**Estimate:** 2h
**Depends on:** 32.1 (build tooling)

- Add a proper 4D Hilbert curve preset to `LSystemPresets` (uses `>`,`<`
  ana-axis rotation symbols in the grammar)
- Generate reference image via integration test
- Add to both `integration-tests.sh` and `manual-test.sh`

---

### Task 32.11: Enum-Typed Builder Dispatch — A4

**Estimate:** 2h
**Priority:** Low
**Depends on:** 32.7 (dispatch unification — benefits from the context)
**Finding:** A4 (from Sprint 31 architecture review)

**Problem:** `LSystemSceneBuilder.resolveSubBuilder` dispatches on `"curve"`, `"sphere"`,
`"cone"` strings with a wildcard default. No compile-time safety.

**Implementation:**
- Introduce a sealed trait/enum for sub-builder types
- Change `emitRun` to tag ObjectSpecs with the enum instead of raw strings
- Convert `resolveSubBuilder` to a pattern match on the enum
- Remove the wildcard default

**Done when:** Mis-spelling a builder type in `emitRun` fails at compile time.

---

### Task 32.12: Move 4D Presets into LSystemPresets — A5

**Estimate:** 1h
**Priority:** Low
**Depends on:** 32.1 (build tooling)
**Finding:** A5 (from Sprint 31 architecture review)

**Problem:** 3D presets live in `menger.objects.LSystemPresets`; 4D presets (`HilbertCurve4D`,
`Tree4D`) are hardcoded in `LSystemTurtle4D` companion object.

**Implementation:**
- Add 4D preset entries to `LSystemPresets` (or a separate `LSystemPresets4D`)
- Update `LSystemSceneBuilder.generateFromSpec` to look up 4D presets when `dim=4`
- Make the preset lookup interface consistent: `LSystemPresets(name)` returns the
  same `(axiom, rules, angle, segLen, initWidth, decay, defaultIters)` tuple

**Done when:** Adding a new 4D preset touches the same file as adding a 3D preset.

---

## Task Dependency Graph

```
32.1 (sbt + hygiene + ADR) ── must go first (sbt impacts all)
  │
  ├─► 32.2 (Cauchy IOR) ──► 32.3 (hero-wavelength) ──┬──► 32.4 (denoiser interplay)
  │                                                    ├──► 32.5 (presets/demos)
  │                                                    └──► 32.6 (tests/docs)
  │
  ├─► 32.7 (T1 dispatch unification) ──► 32.11 (A4 enum dispatch)
  │
  ├─► 32.8 (T2+T9+T7 quality infra) ── T9 benefits from 32.7
  │
  ├─► 32.9 (T5+T10 arch hardening) ── independent
  │
  ├─► 32.10 (4D Hilbert preset) ── independent, quick
  │
  └─► 32.12 (A5 preset unification) ── independent, quick
```

## Parallelization Opportunities

| Block | Tasks | Rationale |
|-------|-------|-----------|
| Block A | 32.2 → 32.3 → 32.4, 32.5, 32.6 | Dispersion track — sequential within, independent of architecture |
| Block B | 32.7 → 32.11 | Dispatch unification — same code area |
| Block C | 32.8 | Quality infra — independent files |
| Block D | 32.9 | ArchUnit + fast-path — different concerns |
| Quick wins | 32.10 + 32.12 | ~3h, slot anywhere after 32.1 |

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 32.1 | sbt 2.0.1 + CODE_IMPROVEMENTS cleanup + T11 ADR | 6h |
| 32.2 | Dispersion material parameter + Cauchy IOR model | 4h |
| 32.3 | Hero-wavelength sampling in shaders | 8h |
| 32.4 | Accumulation + denoiser interplay | 3h |
| 32.5 | Dispersion presets + demo scenes | 2h |
| 32.6 | Dispersion tests + reference images + documentation | 5h |
| 32.7 | T1 — Unify object-type dispatch | 16h |
| 32.8 | T2 (PerfCheck) + T9 (script parity) + T7 (determinism) | 17h |
| 32.9 | T5 (ArchUnit) + T10 (fast-path guard) | 10h |
| 32.10 | 4D Hilbert curve preset + ref image | 2h |
| 32.11 | A4 — Enum-typed builder dispatch | 2h |
| 32.12 | A5 — Move 4D presets into LSystemPresets | 1h |
| **Total** | | **~76h** |

---

## Definition of Done

- [ ] All success criteria met (dispersion + architecture)
- [ ] Pre-push hook green — including unchanged references for dispersion=0
- [ ] CHANGELOG.md updated
- [ ] Architecture backlog items T1, T2, T5, T7, T9, T10, T11, A4, A5 marked done
- [ ] ARCHITECTURE_REVIEW.md updated with closure notes
- [ ] Integration + manual test scripts cover dispersion + 4D Hilbert scenes
