# Sprint 33: Production-Quality Caustics (Physically Validated)

**Sprint:** 33 - Production-Quality Caustics
**Status:** 🔄 In Progress — scope rewritten 2026-07-02; **course-corrected 2026-07-03** (see below)
**Estimate:** ~77 hours (range 65–85)
**Branch:** `feature/sprint-33`

> ### ⚠️ Course correction (2026-07-03) — architecture ground-truth was wrong
>
> Empirical investigation (strace + md5 + real optix-jni tags) overturned three premises of
> the 2026-07-02 plan. Corrections, all verified:
>
> 1. **The live caustics shader is optix-jni's, not menger-geometry's.** optix-jni's
>    `optix_shaders.cu` `#include`s `caustics_ppm.cu` (+ `hit_menger4d.cu` + `hit_curve.cu`),
>    producing the **one monolithic `optix_shaders.ptx`** that `PipelineManager` loads for
>    **every** scene. menger-geometry's `optix_shaders_menger.ptx` is an incomplete (no curve
>    programs) duplicate that is **never loaded**. ∴ caustics physics fixes (P1–P9) must land
>    in **optix-jni's** `caustics_ppm.cu`. The Decision-6 "move orchestration to
>    menger-geometry" layering is deferred to backlog (it also wouldn't de-Menger optix-jni,
>    which still owns menger4d + curve). *(user decision 2026-07-03: "physics + binding fix
>    only")*
> 2. **Release is 0.1.12, not 0.1.11.** 0.1.11 is already published as a PBR-texture release.
>    Caustics batches into **optix-jni 0.1.12** (P1–P9 + the `getCausticsStats` FindClass fix
>    + the P4 linear-composite seam). menger `build.sbt` pins `0.1.12-SNAPSHOT` during dev.
> 3. **Whole-image L3 (C8) against pbrt `sppm` is not achievable** and must change to a
>    **caustic-delta metric**. pbrt has full global illumination; menger is direct + caustics
>    only, and menger's diffuse model is non-physical (`0.3` constant ambient + `0.7` direct
>    blend), so the direct-lit floor sits ~14% high and GI-dominated regions differ
>    structurally — none of which caustic fixes touch. Validation switches to comparing the
>    **caustic contribution in isolation** (`caustics-on − caustics-off`), which cancels
>    ambient + direct. The physical-lighting rewrite (drop ambient/blend) is backlogged as
>    **F-PBR-DIFFUSE** (regenerates every reference image). *(user decisions 2026-07-03:
>    "diagnose floor/units first" → "delta metric only; backlog the lighting rewrite")*
>
> **Progress under the correction:** P1 (emission measure) + P6 (iteration normalization) +
> the FindClass fix are applied to optix-jni's live shader and **proven live** (0.1.12-SNAPSHOT
> published, menger repinned, strace-confirmed): canonical-scene mean brightness moved from
> **+68% → +31%** vs pbrt. Tasks 33.3–33.6 are re-scoped around this reality below; the task
> bodies still describing the menger-geometry shader as "live" are superseded by this box.
**Dependencies:** Sprint 32 (spectral machinery enables dispersive caustics, task 33.10).
The current PPM implementation (`menger-geometry/.../shaders/caustics_ppm.cu`) is the input
state — kept as the algorithm, its physics rebuilt.
**Feature ID:** F16 in [FEATURE_DEPENDENCIES.md](FEATURE_DEPENDENCIES.md); T6 (caustics
ladder) from architecture backlog — C1–C7 implemented here, C8 gated against pbrt-v4.

---

## Goal

Deliver **correct, production-quality caustics for arbitrary geometries**, validated against
pbrt-v4. The previous plan ("conclude the PPM parameter investigation, lock defaults") rested
on a false premise: the implementation has structural physics bugs, so no parameter tuning can
reach correctness. The long tuning saga (magic scales 0.03×…10000×, brightness stuck at 38–54%
of pbrt) is fully explained by the defect list below.

This sprint fixes the physics (emission pdf, Fresnel/reflection, linear compositing, density
estimate, direct-light double counting), generalizes emission from a single sphere target to
arbitrary refractive geometry, adds dispersive **and** reflective caustics, and builds a
**layered validation pyramid** so correctness is enforced, not asserted.

### Validation pyramid

Bytewise pbrt-vs-menger equality is impossible (different Monte-Carlo samplers/RNG produce
different per-pixel noise). Instead:

- **L1 — Analytic (CPU, no GPU):** Snell, exact Fresnel, focal point, emission pdf/power
  closed forms. Ladder rungs C1, C3, C4.
- **L2 — Statistical (GPU):** energy conservation ±5%, PPM convergence, brightness ratio,
  hit-rate vs geometric cross-section. Ladder rungs C2, C5, C6, C7. Driven through the
  already-plumbed `CausticsStats`.
- **L3 — Converged-reference (pbrt-v4):** menger vs pbrt in **linear space** (menger gains a
  minimal PFM float dump), quantitative metrics (`imgtool` MSE/FLIP), fixed seeds, thresholds
  locked from measured converged results. Ladder rung C8. **Course-corrected (2026-07-03):**
  the metric compares the **caustic contribution in isolation** (`caustics-on − caustics-off`
  on each side), because whole-image comparison is dominated by pbrt's global illumination and
  menger's non-physical ambient — neither of which caustic physics affects (see the correction
  box above; F-PBR-DIFFUSE backlog).
- **Bytewise** is reserved for menger-vs-menger determinism (task 33.12).

---

## Success Criteria

- [ ] C1–C7 caustics ladder tests pass (analytic + statistical); C8 passes against a
      committed, manifest-locked pbrt-v4 reference on all twin scenes
- [ ] `--caustics` with **no** further parameters produces correct results on every ladder
      scene (auto-tuning derives photon budget, radius, iterations from the scene)
- [ ] Caustics render correctly for arbitrary refractive geometry (multiple glass objects,
      off-center scenes, glass cube), not just a single centered sphere
- [ ] Reflective caustics render (photon reflection at dielectric/metal boundaries)
- [ ] Dispersive caustics: white light through a dispersive glass sphere produces a
      rainbow-fringed caustic; non-dispersive scenes remain bit-compatible
- [ ] Caustic radiance is composited in **linear space** before the global tone map; works
      under `--tonemap none|reinhard|aces`
- [ ] Caustics off by default; existing non-caustics reference images byte-identical
- [ ] All tests pass; pre-push hook green

---

## Verified defects (root causes of the failed tuning investigation)

Shader: `menger-geometry/src/main/native/shaders/caustics_ppm.cu`.

| ID | Defect | Location |
|----|--------|----------|
| P1 | Emission pdf ignored — photons carry `intensity/N`, but emission importance-samples a cone/disk; missing solid-angle/area measure factor | `calculatePhotonFlux` :678, `emitPointPhoton` :629, `emitDirectionalPhoton` :582 |
| P2 | Fresnel-reflected energy discarded — photon always refracts weighted `(1−F)`; no reflective caustics | `__closesthit__photon` :804 |
| P3 | Schlick approximation instead of exact dielectric Fresnel | `__closesthit__photon` :799 |
| P4 | Non-physical composite — private exponential tone map + screen blend into the 8-bit LDR buffer | `__raygen__caustics_radiance` :1018–1058 |
| P5 | Density estimate wrong — spurious cosθ weight, unnormalized Gaussian (~8× undercount), missing Lambertian ρ/π floor albedo | `depositPhoton` :352–414, radiance :997–1016 |
| P6 | No cross-iteration normalization — brightness scales ~linearly with iteration count | `__raygen__caustics_radiance` :999–1016 |
| P7 | Single-target emission — arbitrary/multi-object scenes unsupported at emission | `caustic_target_center/radius`, `__raygen__photons` |
| P8 | Grid bounds hardcoded ±3 — off-center scenes get no caustics | optix-jni `OptiXWrapper.cpp` :1804–1809 |
| P9 | Direct-light double counting — every photon deposited, including ones that never touched glass (should store only LS⁺D paths) | `__miss__photon` :834–854 |

---

## Architecture notes (load-bearing)

- **Two copies of the caustics native code.** The live *shader* is menger-geometry's
  `caustics_ppm.cu`; the live *orchestrator* was optix-jni's `CausticsRenderer.cpp` — the
  Sprint-25 injection seam (`ICausticsRenderer`/`setCausticsRenderer`) was never wired, so
  menger-geometry's orchestrator copy is dead. **Decision (2026-07-02):** the orchestrator is
  feature logic and belongs in menger-geometry. Task 33.6 finishes the injection so
  menger-geometry's `CausticsRenderer` becomes the single live orchestrator; optix-jni keeps
  only the data contract + rendering seams and deletes its own orchestrator copy.
- **No linear HDR pipeline exists today.** Tone mapping is applied in-shader and quantized to
  8-bit in the ray payload. The new PFM output is therefore 8-bit-quantized linear clamped to
  [0,1]; the pbrt side applies the identical clamp on EXR→PFM, so L3 comparison is
  apples-to-apples. A true float-HDR film buffer is a backlog item.
- **Struct/host/base-shader changes require an optix-jni release.** All such changes batch
  into **one 0.1.11 release** (task 33.6). Shader-only fixes (33.3–33.5) land in menger first
  and are verified via the **L3 pbrt harness + L1 CPU suite** in the interim.
- **`getCausticsStats` is broken in optix-jni ≤ 0.1.10** (`JNIBindings.cpp` binds
  `getCausticsStatsNative` to the nested class name `io/github/lene/optix/OptiXRenderer$CausticsStats`,
  but the artifact ships a **top-level** `case class CausticsStats` → `NoClassDefFoundError`).
  The L2 statistical suite (`CausticsStatsSuite`) therefore **cancels** its structural rungs
  until the FindClass path is fixed in the 0.1.11 release (Task 33.6). Until then, 33.3–33.5
  are validated by L3 (pbrt) + L1 (CPU), not by `CausticsStats`.

---

## Tasks

### Task 33.1: Validation harness v1 — pbrt pipeline + PFM output + baseline
**Estimate:** 8h
**Depends on:** — (gates everything; do first)

- `scripts/caustics-validation/`: `scenes/` (hand-authored pbrt twins, seeded with
  `canonical-caustics.pbrt` copied from the optix-jni repo); `render-pbrt-references.sh`
  (manual/CI only, never in hooks; `pbrt --seed 0`; convergence gate via a 2×-budget gold
  render and `imgtool diff --metric MSE`; `manifest.txt` records pbrt version, seed, spp,
  scene SHA); `compare-caustics.sh <menger.pfm> <ref.pfm>` (imgtool MSE + FLIP; SSIM via
  ImageMagick `compare -metric SSIM`; per-scene thresholds file).
- Committed references live in `scripts/reference-images/caustics/` (PFM + PNG + manifest).
- PFM writer in `menger-app/.../engines/ScreenshotFactory.scala`, triggered by
  `--save-name foo.pfm` (values = byte/255; documented quantized-linear caveat).
- Baseline capture against the current (broken) implementation; record MSE/FLIP/SSIM as the
  "before" row in `CAUSTICS.md`. Every physics task must move these numbers and cite them.
- Pin the light-unit convention (pbrt point `I` = W/sr vs menger `Light.intensity`) in the
  harness docs **before** P1 brightness is judged.
- Close `CAUSTICS_ITERATION_LOG.md` with a pointer to the defect list.

**Done when:** `compare-caustics.sh` produces a reproducible metrics row for the canonical
scene against a committed, manifest-locked pbrt reference.

---

### Task 33.2: Test skeleton — analytic C1–C4 (CPU) + statistical C5–C7 (GPU)
**Estimate:** 8h
**Depends on:** 33.1 (shared canonical-scene constants)
**Status:** ✅ Done (2026-07-03)

New package `menger-app/src/test/scala/menger/caustics/`:

- `CausticsPhysicsSuite` (CPU, `AnyFlatSpec`, normal test tier): C1 cone-pdf/power closed
  forms; C3 Snell exit angles < 0.01 rad + TIR at asin(1/1.5); C4 paraxial focal point ±0.1;
  exact Fresnel values (F(0°,1.5)=0.04, Brewster, grazing, R+T=1). **12 tests green.**
- `CausticsStatsSuite` (single GPU suite, `taggedAs GPURequired`, headless render +
  `getCausticsStats`, small budget): C1 count/flux, C2 hit rate, C5 conservation, C6 radius
  monotonicity + P6 iteration-invariance, C7 peak/ambient. Consolidated into one suite rather
  than four (fewer files; same scene fixture).
- Structural rungs currently **cancel** because `getCausticsStats` is broken in optix-jni
  ≤ 0.1.10 (FindClass mismatch, unblocked in Task 33.6); physics rungs are **pending**, tagged
  by the defect ID they wait on (P1→33.3, P6→33.3, P2/P9→33.4, P5→33.5). Each physics task
  flips its rungs green. The GPU render itself executes today (`MengerRenderer` loads the JNI
  symbols from `libmengergeometry.so`); only the stats readback is blocked.

---

### Task 33.3: P1 + P6 — emission power/pdf and per-iteration normalization
**Estimate:** 4h
**Depends on:** 33.2
**Status:** ✅ Done (2026-07-03) — applied to **optix-jni's** live `caustics_ppm.cu` (not the
menger-geometry dead copy; see correction box), bundled with the `getCausticsStats` FindClass
fix into `optix-jni 0.1.12-SNAPSHOT`. Validated live: canonical mean brightness +68% → +31% vs
pbrt (whole-image; caustic-delta metric lands in the 33.1 harness extension).

- `caustics_ppm.cu`: `calculatePhotonFlux` takes an emission-measure factor from the emit
  helpers (point: `I·2π(1−cosθmax)/N`; directional: `E·π·r_disk²/N`); `__raygen__caustics_radiance`
  divides accumulated τ by `iterations`. Fix the half-right comment at :999–1003 with the full
  derivation. Keep the Hachisuka τ-ratio scaling in `__raygen__update_radii` (:947–953).
- Delete magic-scale history.

**Proof:** C1 flux total green; iteration-invariance green; harness brightness delta recorded.

---

### Task 33.4: P3 + P2 + P9 — exact Fresnel, RR reflection, specular-only deposition
**Estimate:** 6h
**Depends on:** 33.3

- `__closesthit__photon`: exact dielectric Fresnel helper (replaces Schlick + TIR check);
  Russian-roulette reflect/refract with probability F, flux **unweighted** (Beer-Lambert on
  exit unchanged); RNG seed into photon payload; "touched specular" bit set on interaction;
  `__miss__photon` deposits only when the bit is set. Add a `reflection_events` counter (struct
  change rides the 33.6 release; until then assert via refraction/emitted ratio).
- Reflective caustics fall out free.

**Proof:** CPU Fresnel values; C5 conservation ±5% and reflect:refract ≈ F̄; new
`reflective-caustic.pbrt` twin shows the reflected arc in both renderers; zero caustic delta
outside the ring (P9 regression via image subtraction).

---

### Task 33.5: P5 — density estimate: kernel normalization, cos removal, floor albedo ρ/π
**Estimate:** 4h
**Depends on:** 33.4

- `depositPhoton`: remove the cosθ weight and the unnormalized Gaussian → uniform-disk deposit
  τ += Φ (pbrt-matching kernel).
- `__raygen__caustics_radiance`: L = (ρ/π)·τ/(π r²·iterations); floor albedo ρ captured into the
  unused `HitPoint.weight` at hit-point init (share the checker/solid-color logic with
  `miss_plane.cu`; helper extraction rides 33.6, interim local duplicate).

**Proof:** C7 absolute bound from analytic focal-spot power; harness MSE step-change (~8×).

---

### Task 33.6: P4 — linear-space compositing + injection wiring + optix-jni 0.1.11
**Estimate:** 12h
**Depends on:** 33.3–33.5
**Status:** ✅ Done (2026-07-03) — P4 physics fix landed: `__raygen__caustics_radiance` drops
the private exponential tone map + screen blend and additively composites linear radiance
through the single global tone-map operator (optix-jni `1098074`). The **layering move**
(injection wiring + deleting optix-jni's dead `CausticsRenderer.cpp`) is **deferred to backlog**
per user decision "Physics + binding fix only" — the caustics release is 0.1.12, not 0.1.11.
Caustic-delta vs pbrt: spatial correlation 0.11 → 0.86 (> 0.8 target).

The one cross-repo task: a single change-set in `/home/lene/workspace/optix-jni` released as
0.1.11, then bump `build.sbt`.

- **optix-jni (data contract + seams only):** `OptiXData.h` — `float4* caustic_radiance` in
  `BaseParams`, `reflection_events` in `CausticsStats`, `CausticTarget{center,radius}` list in
  `CausticsParams` (for 33.7). `BufferManager` allocates/zeros the new buffers. `miss_plane.cu`
  (and `hit_plane.cu` if planes shade there too): primary rays (depth == 0) add
  `caustic_radiance[pixel]` to the linear color **before** the tone-map block. Widen
  `ICausticsRenderer` so the orchestrator runs its passes without `OptiXWrapper` privates; wire
  the injection call path; **delete optix-jni's `CausticsRenderer.cpp`**. Replace the hardcoded
  grid bounds with plumbing for a hit-point AABB (33.7 supplies the algorithm).
  **Fix `JNIBindings.cpp` `getCausticsStatsNative`**: `FindClass` must use the top-level
  `io/github/lene/optix/CausticsStats`, not the nested `OptiXRenderer$CausticsStats` — the
  current mismatch aborts the whole L2 `CausticsStatsSuite` (`NoClassDefFoundError`). This
  unblocks the C1/C2/C5/C6/C7 structural rungs.
- **menger-geometry:** its `CausticsRenderer` becomes the live orchestrator via injection;
  reorder passes to hit points → grid → photons → **radiance → final render**.
  `__raygen__caustics_radiance` deletes the private tone map, screen blend and byte writes, and
  writes linear L into the `caustic_radiance` buffer.
- arc42: module-description update + §9 decision record (orchestration in menger-geometry,
  contract in optix-jni).

**Proof:** C7 measured in linear PFM; tone-map matrix integration test (none/reinhard/aces ×
caustics on); existing non-caustics references byte-identical (caustics off ⇒ buffer never
read); caustics+denoise smoke render.

---

### Task 33.7: P7 + P8 — arbitrary-geometry emission, dynamic grid, hit-point correctness, PLY exporter
**Estimate:** 10h
**Depends on:** 33.6
**Status:** ✅ Done (2026-07-03, core) — **P7** arbitrary-geometry emission is already covered by
the host-computed merged bounding sphere over all refractive instances (OptiXWrapper.cpp ~1794);
per-instance ΔΩ-weighted target selection is a documented future optimization for overlapping
multi-object scenes. **P8** dynamic grid bounds landed (optix-jni `3801248`): grid centred on the
emission target, sized by target radius instead of the ±3 hardcode. Validated no regression
(correlation 0.86). **Deferred to backlog:** hit-point `optixTrace`-result correctness and the
PLY exporter (menger-side, not required for the 0.1.12 release).

- **Emission targeting:** host emits one bounding sphere per refractive instance (ior > 1.05)
  instead of one merged AABB; per photon pick target i with probability ΔΩ_i/ΣΔΩ, sample its
  cone; Φ = color·I·ΣΔΩ/N (partition; overlaps double-count — documented approximation). Keep
  the merged-AABB fallback and the legacy non-IAS `params.sphere_ior` path working.
- **Grid bounds:** atomic min/max of hit-point coordinates on device; `buildGrid` reads them
  back, sets bounds and `cell_size = max(mean radius, extent/GRID_RES)`. Kill the ±3 hardcode.
- **Hit-point correctness:** honor the actual `optixTrace` result in `__raygen__hitpoints` (only
  create a hit point when the nearest hit *is* the plane). planes[0]-only stays a documented
  limitation.
- **PLY exporter:** `MengerMeshExporter` (or `--export-ply`) dumps the uploaded triangle mesh so
  cube/sponge scenes get pbrt twins (`Shape "plymesh"`). Add a `glass-cube.pbrt` twin +
  reference.

**Proof:** two-spheres twin → two rings in both renderers; glass-cube twin comparable; off-center
scene (plane at y=−5) renders a caustic (fails today with ±3 bounds); per-target C2 stats.

---

### Task 33.8: C8 gate + default parameters + auto-tuning
**Estimate:** 8h
**Depends on:** 33.3–33.7
**Status:** ✅ Done (2026-07-04, core) — shipped in optix-jni 0.1.13. Auto gather radius derives
from the refractive geometry's bounding radius (`CAUSTICS_AUTO_RADIUS_FACTOR · target_radius`)
when unset; `MENGER_CAUSTICS_RADIUS` env knob added for calibration. **C8 locked** as the
caustic-delta spatial correlation ≥ 0.80 (canonical 0.859) in `thresholds.txt` — NOT whole-image
SSIM/MSE, which is structurally loose (menger is primary-ray-only + non-physical ambient →
~5× caustic energy deficit at every radius; documented as F-PBR-DIFFUSE / F-CAUSTICS-SDS).
Calibration finding: correlation is flat across radius, factor 0.6 lands the canonical scene at
its best radius. **Deferred:** photon-budget ΣΔΩ derivation + iteration/accumulation coupling
(defaults are sane; energy is structural-limited so photon count won't change magnitude).

- Calibrate and lock: initial radius r₀ = k·bboxDiagonal; photon budget from ΣΔΩ per light
  (lights that can't see glass get zero budget); iterations tied to
  `RenderSettings.accumulation`; α default calibrated (0.7 vs pbrt SPPM's 2/3). All overridable;
  `--stats` reports the derived values.
- Lock C8 thresholds from measured converged results (target SSIM > 0.90; achieved MSE/FLIP ×
  1.5 as the regression floor). Consider masking the sphere-silhouette region (caustics not
  visible through glass — known limitation).
- Update `CAUSTICS_TEST_LADDER.md` status; record locked defaults in `CAUSTICS.md` + arc42 §9.

**Done when:** bare `--caustics` passes C8 on all committed twin scenes.

---

### Task 33.9: CLI/DSL surface finalization
**Estimate:** 3h
**Depends on:** 33.8
**Status:** ✅ Done (2026-07-04) — shipped the `None`=auto radius surface. `menger-common`
0.1.5 relaxes `CausticsConfig`'s `require(initialRadius > 0)` to accept `0.0`
(`CausticsConfig.AutoRadius`) as the auto-derive sentinel (binary-compatible; optix-jni stays
on its own pin, sbt evicts to 0.1.5). `Caustics.initialRadius` is now `Option[Float]`
(`None` = auto; presets `Default`/`HighQuality` keep an explicit `Some(1.0f)`, so no reference
drift). CLI `--caustics-radius` lost its default → unset = auto (verified pixel-identical to the
old 0.1 default on the `caustics minimal` smoke reference at its 1000-photon budget). Added
scene-precondition warnings (caustics enabled + no refractive object / no lights → `logger.warn`,
never fatal) in `SceneConverter`.

Scope trim (ponytail): only the **radius** gets `None`=auto because only the radius has a native
auto-derive path (optix-jni 0.1.13). Photon/iteration auto-derivation is not implemented natively,
so wrapping them in `Option[Int]` would be indirection with no behaviour — left as `Int`, tracked
in F-CAUSTICS-AUTO-CLI (`docs/BACKLOG.md`).

- `Caustics(enabled, photonsPerIteration, iterations, initialRadius: Option[Float] = None, alpha)`
  — `None` = auto; same optionality on the CLI (`--caustics-radius` unset = auto). Warns (doesn't
  fail) when caustics are enabled with no refractive object or no lights. Extended
  `CausticsCLIOptionsSuite` and `menger.dsl.CausticsSuite`.

---

### Task 33.10: Dispersive caustics
**Estimate:** 5h
**Depends on:** 33.4, Sprint 32 (hero-wavelength + Cauchy IOR)
**Status:** ✅ Done (2026-07-04) — shipped in optix-jni 0.1.13. Photons carry a hero wavelength
(payload p10), refract with the Cauchy `n(λ)=a+b/λ²` for dispersive instances, and deposit flux
tinted by the wavelength's CIE response → spectral floor caustics. Non-dispersive scenes are
bit-identical (canonical caustic-delta correlation unchanged at 0.859); a dispersive glass sphere
gains a coloured floor caustic (saturation 0 → 0.044).

- Sample each photon's λ with the Sprint 32 stratification (payload seed from 33.4); refraction
  through dispersive instances uses n(λ) (Cauchy, per-instance dispersion); photon RGB energy =
  CIE response for λ (same conversion as camera-side hero wavelengths).
- Non-dispersive scenes: λ never alters refraction, RGB sums to white — assert bit-compatible
  with 33.8 results (regression in `CausticsConvergenceSuite`).
- Reference scene: white light through a dispersive sphere → rainbow-fringed ring; pbrt spectral
  twin for **qualitative** comparison only (spectral models differ); gate on menger-only
  invariants (hue ordering across the ring radius) + a manual-test entry.
- Auto-tuning multiplies the photon budget ~4× for dispersive scenes.

---

### Task 33.11: Reference ladder → integration suite + documentation
**Estimate:** 4h
**Depends on:** 33.8–33.10
**Status:** 🔶 Partial (2026-07-09) — arbitrary-geometry test coverage landed; the canonical-ladder
CI wiring and docs consolidation below are still open. Added `test_torus_caustics()` and
`test_icosahedron_caustics()` smoke tests to `scripts/integration-tests.sh` (fast, low photon
budget matching `test_caustics`'s tier — execution + committed-reference-diff only, no pbrt
ground truth, since no PLY-exporter/pbrt-twin exists for arbitrary meshes yet) plus matching
`scripts/manual-test.sh` entries. Icosahedron uses the built-in Platonic-solid CLI path
(`--objects type=icosahedron:material=glass --caustics`), not a new DSL scene or new mesh
infrastructure — confirmed via direct high-photon-budget testing (500k×20, matching
`CausticsCanonical`'s budget) that the arbitrary-geometry emission targeting from Task 33.7
genuinely produces a visible caustic on a 20-facet polyhedron, not just spheres/tori; this had
never been exercised before. **Deferred (not done):** the `test_caustics_ladder()` +
`compare-caustics.sh` canonical/two-spheres CI wiring, and the docs consolidation — both bullets
below remain open.

- `scripts/integration-tests.sh`: keep the fast smoke test; add `test_caustics_ladder()` calling
  `compare-caustics.sh` for **two** scenes (canonical + two-spheres) at 400×300 with reduced
  photon budget against dedicated committed references (keeps pre-push wall-time bounded;
  full-ladder + 800×600 comparisons stay in `compare-caustics.sh --full`, manual/CI). The pbrt
  binary is **never** invoked by hooks — only committed references are read.
- Verify pre-push rendering-path gating covers `caustics-validation/` and `caustics_ppm.cu`.
- `scripts/manual-test.sh`: caustic + dispersive-caustic scenes appended.
- **Multi-object caustics — menger verification (REQUIRED before considering the feature done):**
  the optix-jni `7e6e594` multi-object caustics fix (per-instance photon emission) ships in
  optix-jni **0.1.15** with a unit test on the optix-jni side. It must **also** be verified on
  the menger side once the pin is bumped to 0.1.15:
  - `scripts/integration-tests.sh`: a two-sphere (N ≥ 2 separated refractive objects) caustics
    scene asserting **two distinct, correctly-placed** caustics (regression guard — the old
    merged-target code produced one misplaced blob).
  - `scripts/manual-test.sh`: an interactive multi-object caustics scene for human visual
    confirmation (per repo policy: every rendering feature needs both a headless regression and
    a manual-test entry).
  - Either light type works: 0.1.15 per-instance emission is implemented in **both**
    `emitPointPhoton` and `emitDirectionalPhoton` (directional was ported before release).
    optix-jni `MultiObjectCausticsSuite` covers both (two-vs-one refraction rate; the old merged
    target collapsed it — point to < 10%, directional to ~13%). Single-object emission is
    bit-identical to before (CDF draw guarded on N > 1), so existing single-object caustic
    references are unaffected.
- Docs: consolidate `docs/caustics/` to `CAUSTICS.md` + `CAUSTICS_REFERENCES.md` + the (now
  enforced) ladder doc; delete resolved analysis docs per repo policy; CHANGELOG.md entry;
  user-guide Caustics section; arc42 §9/§10.

---

### Task 33.12: Fix RenderDeterminismSuite (GPU)
**Estimate:** 2h
**Finding:** Sprint 32 architecture review (MEDIUM)
**Status:** ✅ Done (2026-07-04) — the GPU rung now renders an opaque sphere scene twice
(AA off, single hard-shadow sample, no caustics) and asserts byte-identical `render()` output
(4 succeeded, 0 pending). Caustics excluded and documented in the suite (float-atomic gather is
order-nondeterministic). arc42 §10 gains reliability scenario R6 backing the claim.

`RenderDeterminismSuite` GPU render-determinism test is `pending` — never executes. arc42 §10
reproducibility claims are not backed by an active fitness function.

- Implement byte-identical render comparison: render the same scene twice with a fixed seed,
  assert PNG byte equality (`--seed 42 --allow-uniform-render`).
- **Addition:** the caustics flux path uses float `atomicAdd` (order-nondeterministic) —
  exclude caustics scenes or use tolerance there; document in the suite.

---

### Task 33.13: Expand PerfCheck baseline coverage
**Estimate:** 2h
**Finding:** Sprint 32 architecture review (MEDIUM)
**Depends on:** 33.7 (RR + multi-target changes photon-pass cost)
**Status:** ✅ Done (2026-07-04) — `benchmark.sh` now covers 9 scenes (glass sphere, diamond,
menger4d L2, sierpinski4d L2, tesseract, curve, lsystem tree L3, IBL sphere, **caustics glass**);
`perf-baseline.json` holds real dev-GPU medians (no 5000 ms sentinels). P1 (<5 s) / P2 (<500 ms)
budgets documented in the script header; threshold set to 1.30 (the ~50 ms scenes jitter too much
for a tight ratio — absolute budgets are the primary criterion). Regression check green.

`perf-baseline.json` has 4 entries but only 2 real measurements; two are 5000ms sentinels.

- Run `benchmark.sh` on 8+ representative scenes (glass sphere, diamond, menger4d L2,
  sierpinski4d L2, tesseract, curves, lsystem tree, IBL sphere) plus one caustics scene; replace
  sentinels with real measurements; document P1 (<5s) and P2 (<500ms) budgets.

---

### Task 33.14: ObjectSpec dispersion cleanup
**Estimate:** 0.5h
**Finding:** Sprint 32 code review (LOW)
**Status:** ✅ Done (2026-07-04) — parse error message now says "non-negative" (matches the
`d < 0f` validation) and a `dispersion=33` example added to the `parse()` docstring.

- Fix error message: "must be a positive number" → "must be non-negative".
- Add `dispersion` to the `parse()` method docstring.

---

### Task 33.15: LSystemTurtle4D minimum-points guard
**Estimate:** 0.5h
**Finding:** Sprint 31 CODE_IMPROVEMENTS (LOW)
**Status:** ✅ Done (2026-07-04) — 4D `emitRun` now guards `points.length < 2` (was
`points.isEmpty`), mirroring the 3D turtle; single-point runs no longer emit degenerate curves.

Mirror the 3D turtle's `points.length >= 2` check in the 4D turtle's `emitRun`. Without it,
single-point runs produce degenerate curves (all points identical).

---

## Task Dependency Graph

```
33.1 (harness) ─► 33.2 (tests) ─► 33.3 (P1+P6) ─► 33.4 (P2+P3+P9) ─► 33.5 (P5) ─► 33.6 (P4 + optix-jni 0.1.11) ─► 33.7 (P7+P8) ─► 33.8 (C8 + auto-tune) ─► 33.9 (CLI/DSL) ─► 33.11
                                                        └───────────────────────► 33.10 (dispersive) ────────────────────────────────────────────────────────► 33.11
33.12–33.15 independent (33.13 after 33.7)
```

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 33.1 | Validation harness + PFM output + baseline | 8h |
| 33.2 | Test skeleton C1–C4 (CPU) + C5–C7 (GPU) | 8h |
| 33.3 | P1+P6 — emission pdf + iteration normalization | 4h |
| 33.4 | P2+P3+P9 — Fresnel, RR reflection, LS⁺D deposition | 6h |
| 33.5 | P5 — density estimate kernel + albedo | 4h |
| 33.6 | P4 — linear composite + injection wiring + optix-jni 0.1.11 | 12h |
| 33.7 | P7+P8 — arbitrary-geometry emission, grid, hit-point, PLY | 10h |
| 33.8 | C8 gate + defaults + auto-tuning | 8h |
| 33.9 | CLI/DSL surface finalization | 3h |
| 33.10 | Dispersive caustics | 5h |
| 33.11 | Integration suite + docs consolidation | 4h |
| 33.12 | Fix RenderDeterminismSuite GPU test | 2h |
| 33.13 | Expand perf-baseline.json | 2h |
| 33.14 | ObjectSpec dispersion cleanup | 0.5h |
| 33.15 | LSystemTurtle4D minimum-points guard | 0.5h |
| **Total** | | **~77h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] CHANGELOG.md updated
- [ ] Caustics ladder C1–C7 tests pass; C8 gated against pbrt-v4 reference
- [ ] Caustics ladder scenarios gated in the integration suite
- [ ] optix-jni 0.1.11 released; `build.sbt` bumped
