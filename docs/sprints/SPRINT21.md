# Sprint 21: Higher-Dimensional Fractals

**Sprint:** 21 - Higher-Dimensional Fractals
**Status:** Partially Complete (21.6 Cone/Plane Textures and 21.7 Fog deferred)
**Estimate:** ~27 hours
**Branch:** `feature/sprint-21`
**Dependencies:** Sprint 18 (18.3 — GPU 4D math), Sprint 19 (4D polychora as standalone)

---

## Goal

Implement Menger sponge and Sierpinski tetrahedron analogs in 4D and higher dimensions,
using the GPU 4D math infrastructure from Sprint 18.

## Success Criteria

- [x] 4D Menger sponge analog renders correctly via GPU 4D math
- [x] Higher-dimensional Sierpinski tetrahedron analogs work
- [x] Interactive parameter space exploration (1D, using existing `scene(t)` system)
- [x] Fractional levels work for `sponge-recursive-ias` (visual transition between integer levels)
- [ ] Cone and plane geometry support image textures and PBR maps (normal-map, roughness-map) — deferred to Sprint 22
- [ ] Fog/depth cue renders distance-based attenuation for all geometry — deferred to Sprint 22
- [x] Fractional-level animation demo renders smooth level transition via `scene(t)`
- [x] All tests pass

---

## Tasks

### Task 21.1: 4D Menger Sponge Analog

**Estimate:** 5h
**Depends on:** Sprint 18.3 (GPU 4D math)

A 4D analog of the Menger sponge, using the GPU 4D transform infrastructure.
Rendered as a 3D cross-section at varying 4D rotation angles.

---

### Task 21.2: Higher-Dimensional Sierpinski Tetrahedron Analogs

**Estimate:** 4h
**Depends on:** Sprint 18.3 (GPU 4D math)

Sierpinski tetrahedron analogs in 4D (and higher if feasible), using the same GPU
infrastructure as 21.1.

---

### Task 21.3: Interactive Parameter Space Exploration

**Estimate:** 3h
**Depends on:** 21.1, 21.2

Allow real-time or animation-based exploration of fractal parameters:
- Fractal level (depth)
- 4D rotation angle(s)
- Cross-section position

Scope: 1D exploration using the existing `scene(t)` animation system. Map fractal
parameters to t for sweep animations.

**Deferred:** Multi-dimensional parameter exploration (independently varying 2-3+
parameters) is in the backlog.

---

### Task 21.4: Fractional Levels for IAS Sponge

**Estimate:** 2h  
**Depends on:** Sprint 19.10 spike findings (`docs/dev/sprint-19-spike-fractional-ias.md`)

Implement fractional sponge levels for `sponge-recursive-ias` using Approach B from the
Sprint 19.10 spike: two IAS trees at adjacent integer levels, with the coarser level
fading out.

**Approach (from spike findings):**

For fractional level `L = N + f` (0 < f < 1):
- Tree 1: level `N` at opacity 1.0 (fine structure, fully opaque)
- Tree 2: level `N−1` at transparency `f` → opacity `1−f` (coarser skin, fading out)

At `f=0` Tree 2 is fully opaque → looks like level N−1.  
At `f=1` Tree 2 is fully transparent → looks like level N.  
The transition reveals the finer structure progressively.

**Key facts from spike:**
- No shader or compositor changes needed — overlapping transparent IAS sponges work
  correctly today (`hit_triangle.cu:197`, `accumulateShadowAttenuation` in `helpers.cu`)
- Per-instance alpha already in `ObjectInstance.color[3]`
- Instance budget: two level-N sponges use ~40 of 64 slots — within budget for N ≤ 13

**Implementation steps:**
1. Add `fractionalLevel: Float` overload (or extend existing) to
   `addRecursiveIASSpongeInstance` in `OptiXRenderer.scala:586–732`.  
   When `fractionalLevel` is fractional: call `addRecursiveIASSpongeInstance` twice —
   once for `floor(L)` at opacity 1.0, once for `floor(L)−1` at opacity `1 − frac(L)`.  
   When `fractionalLevel` is integer: single call (existing behavior unchanged).
2. Wire through `TriangleMeshSceneBuilder.scala:104–113` and DSL (`SceneObject` / spec).
3. Add smoke test to `RecursiveIASSpongeSuite`: render two overlapping IAS sponges at
   fractional level, assert no exception and pixel output is non-trivial.

---

### Task 21.5: Documentation

**Estimate:** 2h

- Sprint retrospective
- CHANGELOG.md update
- User guide section on higher-dimensional fractal analogs
- Example renders of 4D fractal cross-sections

---

### Task 21.6: Cone and Plane Image Texture + PBR Map Support

**Estimate:** 4h (CUDA)
**Risk:** Moderate — requires CUDA shader changes and new struct fields
**Resolves:** CODE_IMPROVEMENTS.md M-texture-builder-gap

Cone and plane hit shaders currently support procedural textures but not image textures
or PBR maps (`normal-map`, `roughness-map`). The blocker is that `texture_index` on both
geometry types is repurposed to index the geometry data arrays (`cone_data` / `plane_data`),
leaving no slot for an image texture lookup.

**Implementation:**
1. Add `image_texture_index` field to cone and plane instance data in `OptiXData.h`
   (separate from `texture_index` which remains the geometry data index)
2. `hit_plane.cu`: generate planar XZ UV coordinates; call `sampleInstanceTexture`,
   `applyNormalMap`, `applyRoughnessMap` — follow `hit_sphere.cu` as reference
3. `hit_cone.cu`: generate cylindrical UV coordinates; same calls
4. Wire new field through `OptiXWrapper.cpp` and `JNIBindings.cpp`
5. Expose via `addConeInstance` / `addPlaneInstance` in `OptiXMeshApi.scala` /
   `OptiXPlaneApi.scala`; update scene builders to pass `textureIndex`

**Verification:** Integration tests in `optix-jni/src/test/scala/menger/optix/TextureSuite.scala`
covering cone + plane image texture and PBR maps; reference images committed.

---

### Task 21.7: Fog / Depth Cue

**Estimate:** 4h

Distance-based attenuation in the raygen or closest-hit shader. Makes dense fractal
cross-sections and deep sponge structures legible by fading geometry with distance.

**Implementation:**
1. Add `fog_density: Float` and `fog_color: Vec3` fields to scene/camera params
   (or a top-level `fog { density=... color=... }` DSL block)
2. In `raygen.cu` or `hit_triangle.cu`: apply exponential fog after shading —
   `color = mix(fog_color, shaded_color, exp(-fog_density * hit_distance))`
3. Expose via CLI (`--fog density=0.05:color=0.8,0.8,0.9`) and DSL
4. Smoke test: render with fog enabled, assert pixel values differ from fog-off baseline

**Notes:** No geometry changes. Pure shader addition. Low risk.

---

### Task 21.8: Fractional-Level Sponge Animation

**Estimate:** 3h
**Depends on:** 21.3, 21.4

Demo animation using `scene(t)` to sweep `fractionalLevel` from 1.0 to 4.0,
producing a smooth visual transition through sponge levels. Natural showcase of both
the fractional-level blending (21.4) and the animation system (21.3).

**Implementation:**
1. Ensure `fractionalLevel` is exposed as a `scene(t)`-bindable parameter on
   `sponge-recursive-ias` objects (same wiring as rotation angles in 21.3)
2. Write example scene file `examples/sponge-level-animation.menger` that sweeps level
   1.0 → 4.0 over t ∈ [0,1]
3. Add to user guide as example animation

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 21.1 | 4D Menger sponge analog | 5h | Sprint 18.3 |
| 21.2 | Higher-dim Sierpinski analogs | 4h | Sprint 18.3 |
| 21.3 | Interactive parameter exploration (1D) | 3h | 21.1, 21.2 |
| 21.4 | Fractional levels for IAS sponge | 2h | Sprint 19.10 |
| 21.5 | Documentation | 2h | All |
| 21.6 | Cone/plane image texture + PBR maps (CUDA) | 4h | Sprint 20 |
| 21.7 | Fog / depth cue | 4h | — |
| 21.8 | Fractional-level sponge animation | 3h | 21.3, 21.4 |
| **Total** | | **~27h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] Example renders of 4D fractal cross-sections

---

## Notes

### Task 21.1: Implementation Notes

**Approach:** Custom OptiX intersection shader with iterative IFS traversal — O(1) VRAM regardless of level.
No geometry is stored. Each ray traverses the 4D IFS stack per-thread: project 4D cell bounds to 3D AABB,
fast-reject on miss, recurse to children, intersect projected hypercube faces at leaves.

The "project CPU mesh then upload" path (`TesseractSponge`) cannot scale past level 3 (face count = 24 × 48^N).
This IFS approach matches `TesseractSponge` pixel-exactly at equivalent levels.

**Render times (800×600, one light, matte material, measured on RTX GPU):**

| Level | Frame time | VRAM delta |
|-------|-----------|------------|
| 2     | 58 ms     | flat       |
| 3     | 50 ms     | flat       |
| 4     | 73 ms     | flat       |
| 5     | 188 ms    | flat       |
| 6     | 220 ms    | flat       |
| 7     | 405 ms    | flat       |
| 10    | impractical (not measured) | flat |

VRAM is constant across all levels — confirmed O(1) storage.

**Practical ceiling:** level 7 (~400ms/frame, ~2.5 FPS). Levels 2–4 are real-time (>13 FPS).
Level 8+ increases render time significantly without visual payoff for most use cases.

**rotXW sweep:** 0°→180° produces distinct images at every angle — smooth shape change confirmed.

**GPU 4D Path**

All 4D geometry in this sprint uses the GPU 4D math from Sprint 18.3.
The CPU-side 4D pipeline (`Mesh4D`, `RotatedProjection`) is legacy at this point
and should not be used for new geometry.

### Deferred

- **Multi-dimensional parameter exploration** — backlog (independently varying multiple
  fractal parameters; needs parameter panel or multi-axis input mapping)
- **Fractal subdivision on 4D polychora** (16-cell, 24-cell, 600-cell as subdivision
  bases) — can be added in Sprint 22+ now that standalone polychora exist from Sprint 19
- **L-systems in 3D and 4D** — backlog
- **Rotopes** — backlog
