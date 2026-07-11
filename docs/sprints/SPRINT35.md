# Sprint 35: Visual Quality + Sprint 34 Deferred

**Sprint:** 35 - Visual Quality
**Status:** 📋 Planned — scope confirmed 2026-07-02
**Estimate:** ~42 hours
**Branch:** `feature/sprint-35` (not yet created)
**Dependencies:** None hard. SPRINT34_DEFER items are self-contained.

---

## Goal

Two visual quality features (depth of field, wireframe) plus all 6 Sprint 34 deferred
items: UV infrastructure, uvScale, IOR metadata, metallic/AO tests, convention anchoring,
duplicate texture detection.

---

## Success Criteria

- [ ] `camera { aperture = 0.1, focalDistance = 5.0 }` in DSL produces bokeh blur
- [ ] Depth of field disabled by default (aperture = 0 → pinhole camera, current behaviour)
- [ ] `material { wireframe = true }` renders mesh edges via barycentric detection
- [ ] Cylinder, curve, and 3× 4D hit shaders support PBR texture maps (UV computation)
- [ ] `uvScale` from TextureSetMetadata / ObjectSpec flows to GPU uniform
- [ ] IOR from `menger-textureset.json` overrides material preset IOR
- [ ] Metallic/AO visual diff tests pass (images differ with/without maps)
- [ ] Convention detection fully anchored (no mid-word substring matches)
- [ ] Duplicate texture loading detected and warned
- [ ] F-PBR-DIFFUSE completed: physical Lambertian diffuse shipped, default light model
      reconciled (scenes not near-black), all reference images regenerated
- [ ] All tests pass

---

## Tasks

### Task 35.1: Depth of Field

**Estimate:** 8h

Physically-based depth of field via lens sampling in the raygen shader.

**DSL:**
```scala
case class Camera(
  eye: Vec3 = ...,
  lookAt: Vec3 = ...,
  up: Vec3 = ...,
  fovDegrees: Float = 60f,
  aperture: Float = 0f,       // 0 = pinhole (no DoF)
  focalDistance: Float = 1f,  // world units to focal plane
)
```

**Implementation:**
- In raygen shader: if `aperture > 0`, jitter ray origin on disk perpendicular to view
  direction; adjust direction to converge at `focalDistance`
- Standard thin-lens approximation (PBRT §6.2)
- JNI: `setCameraDoF(aperture: Float, focalDistance: Float)`
- DSL: wire `Camera.aperture` / `Camera.focalDistance` through `SceneConfigurator`
- Ray direction normalization after lens shift

**Integration test:** Sphere scene, aperture=0.05, focal distance at sphere centre —
foreground and background blur visible.

---

### Task 35.2: Wireframe Rendering (barycentric)

**Estimate:** 2h (was 6h cylinder approach — industry standard is barycentric edge detection)

**Research-backed decision (2026-07-02):** Barycentric wireframe is the industry-standard
approach used by Blender, NVIDIA OptiX SDK's `optixCutouts` sample, and most real-time
ray tracers. Detect triangle edges by checking barycentric coordinates in the hit shader.
Zero additional geometry — pure shader-compute decision per hit point.

**DSL:**
```scala
enum RenderMode:
  case Solid      // default
  case Wireframe  // edges only
  case SolidWireframe  // faces + edges overlay

addMesh("model.obj") { renderMode = RenderMode.Wireframe }
```

**Implementation:**
- In closest-hit shader (hit_triangle.cu): after intersection, compute barycentric
  coordinates (u, v, 1-u-v). If min(u, v, 1-u-v) < edgeThreshold → wireframe color
- `edgeThreshold` is a per-material shader parameter (default ~0.02, maps to edge thickness
  in world units at 1-unit distance from camera — same pixel width independent of distance)
- Wireframe color: either material.color with metallic=0 or a dedicated `wireframeColor`
  field on Material
- `RenderMode` stored in InstanceMaterial as `uint render_mode` (0=solid, 1=wireframe, 2=solid+wireframe)
- Performance: barycentric branch adds ~1-2 float compares per hit — negligible

**Integration test:** Cube mesh in wireframe mode, verify edges are visible and faces are
absent (or translucent).

---

### Task 35.3: Integration Tests + Reference Images

**Estimate:** 3h

- DoF scene: integration-tests.sh scenario with aperture blur
- Wireframe scene: cube or icosahedron in wireframe mode
- Add both to scripts/manual-test.sh
- Add both to scripts/integration-tests.sh

---

### Task 35.4: Documentation

**Estimate:** 3h

- User guide: Depth of Field section — aperture, focal distance, noise/accumulation guidance
- User guide: Wireframe Rendering section — render modes, barycentric technique note
- User guide: PBR texture set usage guide (Sprint 34)
- CHANGELOG.md entry

---

### Task 35.5: DEFER-1 — UV Infrastructure for 5 Shaders

**Estimate:** 16h
**From:** Sprint 34 review DEFER-1

Cylinder, curve, and 3× 4D hit shaders (`hit_cylinder.cu`, `hit_curve.cu`,
`hit_menger4d.cu`, `hit_sierpinski4d.cu`, `hit_hexadecachoron4d.cu`) have no UV
computation — they cannot use PBR texture maps at all (no albedo, normal, roughness,
metallic, AO, or height maps).

**Implementation:**
- **Cylinder UV:** `u = atan2(hit.x, hit.z) / (2π)`, `v = hit.y / height` — standard
  cylindrical unwrap
- **Curve UV:** `u = curve parameter t ∈ [0,1]` (already computed for interpolation),
  `v = hit.θ / π` (angular position around curve cross-section). Need to pass t through
  hit shader.
- **4D shaders UV:** Defer decision on 4D UV parameterization. Minimum viable: project
  4D normal to 3D sphere, use spherical UV. Or accept that 4D objects don't get texture
  maps and document the limitation. **Design decision needed.**

Files touched (per shader): hit_*.cu (UV computation + applyNormalMap/applyRoughnessMap/
applyMetallicMap/applyAOMap additions), JNI/InstanceMaterial (render_mode for wireframe
if combined), optix-jni 0.1.12 release.

---

### Task 35.6: DEFER-2 — uvScale GPU Uniform

**Estimate:** 4h
**From:** Sprint 34 review DEFER-2

`ObjectSpec.uvScale` and `TextureSetMetadata.uvScale` are parsed but never consumed.
Need a GPU-side uniform that scales UV coordinates before texture sampling.

**Implementation:**
- Add `float uv_scale` to InstanceMaterial struct (optix-jni)
- Add `float getInstanceUVScale()` accessor in helpers.cu
- In applyNormalMap, applyRoughnessMap, applyMetallicMap, applyAOMap: multiply UV
  coordinates by `uv_scale` before sampling
- JNI: add `uvScale` parameter to all add*Instance methods (or use a separate setter)
- SceneBuilder: pass `spec.uvScale.getOrElse(metadata.uvScale).getOrElse(1.0f)` to
  instance creation
- optix-jni 0.1.12 bump + Maven publish

---

### Task 35.7: DEFER-3 — IOR from Metadata Sidecar

**Estimate:** 2h
**From:** Sprint 34 review DEFER-3

`TextureSetMetadata.ior` is loaded from `menger-textureset.json` but never applied
to materials. The metadata IOR should override material preset defaults.

**Implementation:**
- In TextureManager.loadTextureSet: after loading metadata, apply `metadata.ior` to
  specs that use this texture set (via ObjectSpec.copy?)
- Alternative: pass metadata to SceneBuilder and apply during instance creation
- Precedence: explicit spec.ior > metadata.ior > material preset IOR
- Test: verify `menger-textureset.json` with `{"ior": 2.0}` renders differently from
  default preset IOR

---

### Task 35.8: DEFER-4 — Metallic/AO Visual Diff Tests

**Estimate:** 2h
**From:** Sprint 34 review DEFER-4

No test proves metallic/AO maps change rendered output. The existing integration tests
failed because the test material had metallic=0 (matte preset), so the texture map
multiplying metallic had no effect (0 × any = 0).

**Implementation:**
- Create test data with base material metallic=1.0 (chrome or custom preset)
- Render with metallic map that varies from 0→1 across sphere surface
- Verify images differ with and without metallic map
- Same for AO: use material with color≠black, verify AO darkens at edges
- Add to integration-tests.sh as `test_pbr_visual`

---

### Task 35.9: DEFER-5 — Convention Detection Fully Anchored

**Estimate:** 1h
**From:** Sprint 34 review DEFER-5

Component matching reduced substring false positives but doesn't eliminate them.
A file named `disco_ao.png` still matches `ao` as a component. Fix: require that
the matched component is the ONLY component in the suffix after stripping known
prefix/descriptor parts, or use word-boundary regex instead of component-set matching.

**Implementation:**
- For Poly Haven: match pattern `.*_(diff|rough|metal|ao|disp|nor_gl|nor_dx)\..*` — the
  PBR descriptor must be the LAST underscore-delimited component before extension
- For ambientCG: match pattern `.*_(Color|NormalGL|NormalDX|Roughness|Metalness|AmbientOcclusion|Displacement)_[0-9]+K\..*`
- Update tests: add explicit false-positive prevention test (e.g., `disco_ao.png` alone
  with 2 other non-PBR files → reject, not detect)
- Update TextureSetResolverSuite with explicit anchored-match tests

---

### Task 35.10: DEFER-6 — Duplicate Texture Loading Detection

**Estimate:** 1h
**From:** Sprint 34 review DEFER-6

When an explicit per-map param and a texture-set both reference the same file, the
texture is loaded and uploaded twice — wasting GPU texture slots.

**Implementation:**
- In TextureManager.loadTextures: after collecting staticTextureFilenames, check if any
  texture-set map filenames overlap with static filenames
- If overlap detected: log warning, skip the duplicate static load, reuse the texture-set
  index
- Test: create a spec with both `texture=tiny_diff.png` AND `texture-set=tiny-pbr` (where
  the set also maps `tiny_diff.png`), verify only one upload slot consumed
- No-Render-Impact: loading optimization, output unchanged

---

### Task 35.11: Complete F-PBR-DIFFUSE (physically based Lambertian diffuse)

**Estimate:** 8h
**From:** Sprint 33 — split out of the multi-object caustics release (optix-jni 0.1.15)

The physically based diffuse shader (`albedo/π · irradiance`, no ambient fill) is
**already implemented and validated** against pbrt-v4 (canonical MSE 0.19 → 0.013) but was
held back: it changes the shading of every scene and breaks ~44 GPU tests whose expectations
encode the old `0.3 ambient + 0.7·N·L` model, and it renders default scenes near-black at
unit light intensity. The shader math is correct (root-caused 2026-07-10), not buggy — the
downstream work was deferred by the original commit and never done.

**Code (do not reproduce):** optix-jni commit `72843a0` on branch
`feature/sprint-33-caustics-multitarget`; full handoff in optix-jni
`docs/DEFERRED_PBR_DIFFUSE.md`.

**Implementation:**
- Decide the default look: bump default directional-light intensity (e.g. → π so a fully
  lit surface returns ~albedo) and/or add a small environment/IBL/ambient term so shadow
  sides are not pure black.
- Cherry-pick `72843a0` onto the then-current optix-jni main; reconcile with the chosen
  light model.
- Regenerate all reference images (optix-jni GPU suites + menger integration suite).
- Update GPU-test expectations/scenes (fill light or revised thresholds).
- Re-baseline the pbrt validation harness; release optix-jni; bump menger's pin.

---

### Task 35.12: Fix `CausticsStats.energyConservationError` — not a physical check

**Estimate:** 4h
**From:** Sprint 33 — found while implementing the optix-jni caustics coverage net
(`feature/caustics-coverage-net`, Task 4), 2026-07-11.

**Root cause (investigated, do not re-derive):** `energyConservationError` computes
`abs((deposited + absorbed + reflected) − emitted) / emitted`, intending to verify that a
photon's flux is fully accounted for. This is currently **not a valid comparison** —
`total_flux_deposited` and `total_flux_emitted` are accumulated in incompatible units:

- `total_flux_emitted` (`caustics_ppm.cu` photon-emission raygen, ~line 1148): accumulated
  **once per photon**, via `atomicAdd(&stats->total_flux_emitted, flux_sum)`. This is the
  correct per-photon physical flux.
- `total_flux_deposited` (photon-deposit loop, ~line 397): accumulated **once per
  (photon, matched hit-point) pair**. Each photon does a 3×3×3 grid-cell neighborhood
  search and deposits its *full, undivided* flux into **every** hit point within its
  gather radius (the standard PPM density-estimate technique — this deposit behavior
  itself is correct). With a hit-point grid of thousands of points, a single photon can
  match dozens of hit points, so the raw sum massively over-counts vs. a single photon's
  actual emitted flux.
- The radiance kernel (`__raygen__caustics_radiance`, ~line 1250) *does* correctly
  normalize this for display — dividing by `π·R²·iterations` (see the `P6` comment) to
  convert the raw deposited-flux sum into physical irradiance. That normalization is
  applied only at the point of computing displayed radiance; it is **never** applied to
  the `total_flux_deposited` stats accumulator itself.
- Observed magnitude: for the coverage-net's canonical single-sphere scene,
  `energyConservationError` ≈ **628** (i.e. deposited+absorbed+reflected is ~628× emitted),
  deterministic across repeated identical renders (not noise).
- `total_flux_absorbed` and `total_flux_reflected` (the latter fixed in the same coverage-net
  branch, commit `71d21d5`+`aaf17df` — was previously a dead stat, always 0) *are*
  per-photon and directly comparable to `total_flux_emitted`; the mismatch is isolated to
  `total_flux_deposited`.
- **Blast radius: zero.** No production code and no test other than the coverage net's own
  `CausticsCoverageSuite` reads `energyConservationError`, `fresnelTransmission`, or
  `totalFluxDeposited` (all three derived/raw metrics live in `OptiXRenderer.scala`
  around line 145). Grepped confirmed at time of writing.
- **Interim state:** the coverage-net test (`CausticsCoverageSuite`, "report a bounded,
  deterministic caustic energy-conservation error") recalibrated its ceiling
  (`MaxEnergyConservationErrorRatio`) to the observed ~628× value, documented as a
  **regression guard on the raw ratio**, not a physical conservation claim. This task is
  to make the metric actually mean what its name says.

**Implementation:**
- Decide the normalization point: either (a) normalize `total_flux_deposited` at
  accumulation time (divide by the per-hit-point disk area `π·R²` and by `iterations` at
  the `atomicAdd` site — mirrors the radiance kernel exactly), or (b) track a separate
  "photon-hit-point match count" stat and normalize post-hoc in Scala
  (`OptiXRenderer.scala`). (a) keeps all normalization logic co-located with the existing
  radiance-kernel convention; (b) avoids extra float division in a hot atomic path but
  requires an extra stat.
- After the fix, `energyConservationError` should be small (near 0, allowing for photons
  that escape the gather-radius search or exit the scene unaccounted).
- Re-tighten `CausticsCoverageSuite`'s `MaxEnergyConservationErrorRatio` to the new, small,
  physically meaningful value once fixed — this is now a real regression guard.
- Full optix-jni pre-push gate; verify no reference-image drift (this should be a
  stats-only change, same category as `71d21d5`/`aaf17df`).

---

## Summary

| # | Task | Est |
|---|------|-----|
| 35.1 | Depth of field — lens sampling raygen | 8h |
| 35.2 | Wireframe — barycentric edge detection | 2h |
| 35.3 | Integration tests + reference images | 3h |
| 35.4 | Documentation | 3h |
| 35.5 | DEFER-1: UV infrastructure for 5 shaders | 16h |
| 35.6 | DEFER-2: uvScale GPU uniform | 4h |
| 35.7 | DEFER-3: IOR from metadata sidecar | 2h |
| 35.8 | DEFER-4: Metallic/AO visual diff tests | 2h |
| 35.9 | DEFER-5: Convention detection fully anchored | 1h |
| 35.10 | DEFER-6: Duplicate texture loading detection | 1h |
| 35.11 | Complete F-PBR-DIFFUSE (from Sprint 33) | 8h |
| **Total** | | **~50h** |

---

## Task Dependency Graph

```
35.5 (UV infra) ──┐
35.6 (uvScale)  ──┤── optix-jni 0.1.12 bump
                  │
35.1 (DoF)       ─┤
35.2 (wireframe) ─┤
                  │
35.7 (IOR meta)  ─┤── pure Scala, independent
35.8 (visual diff)┤
35.9 (anchoring) ─┤
35.10 (duplicate) ┘

35.3 (tests)     ← depends on 35.1, 35.2, 35.8, 35.9, 35.10
35.4 (docs)      ← depends on all
```

35.1-35.2 and 35.7-35.10 can run in parallel. 35.5-35.6 block optix-jni bump.
35.3 can start incrementally as each feature lands.

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] CHANGELOG.md updated
- [ ] 6 Sprint 34 deferred items resolved
