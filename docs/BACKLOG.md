# Feature Backlog

Unscheduled feature ideas not yet assigned to a sprint. See [ROADMAP.md](../ROADMAP.md) for
the sprint plan and [CODE_IMPROVEMENTS.md](../CODE_IMPROVEMENTS.md) for tech-debt items.

---

## Features

### F-ND-GEOMETRY: N-Dimensional Geometry (N ≥ 5)

**Priority:** Medium
**Effort:** Large (multiple sprints)
**Dependencies:** Existing 4D projection infrastructure (Sprint 18/25), `menger-geometry` layer

**Description:**
Generalise the 4D→3D projection pipeline to support arbitrary N-dimensional geometry for N ≥ 5.
Each dimension above 3 is projected down one step at a time: 5D→4D→3D, 6D→5D→4D→3D, etc.,
using the same perspective/orthographic projection logic already used for 4D→3D.

**Parameters required per extra dimension (per N > 4):**
- N-D viewpoint: position of the observer in dimension N (analogous to `w` in 4D)
- N-D viewing distance: perspective depth for the projection from N to N-1
- N-D rotation: transformation matrix or angle set operating in dimension N

**Design notes:**
- The projection chain is composable: `projectND → project(N-1)D → … → project4D → project3D`.
  Each step is structurally identical to the existing `Project4D` kernel.
- GPU side: a single generalised kernel parameterised by dimension count, or one kernel per
  projection step chained via intermediate buffers. The former is cleaner; the latter reuses
  existing `Project4D` as-is.
- DSL: `viewpoint5d`, `viewDistance5d`, `rotation5d`, … per extra dimension, or a sequence
  `ndViewpoints: Seq[Float]`, `ndViewDistances: Seq[Float]` indexed by dimension.
- Menger sponge analogs exist in 5D (penteract sponge); same recursive IAS approach applies.
- Memory: each extra dimension adds one projection pass over all vertices; acceptable for
  moderate vertex counts.

**Suggested sprint decomposition (when scheduled):**
1. Generalise `Project4D` CUDA kernel to `ProjectND` parameterised by source dimension
2. Scala DSL + CLI parameters for 5D viewpoint/distance/rotation
3. 5D geometry: penteract (5-cube), 5D Menger sponge analog
4. Integration tests + reference images for 5D scenes
5. Extend to 6D+ (straightforward once 5D pipeline exists)

---

## From PBR Book / pbrt-v4 exploration (Sprint 33)

Ideas noted while studying pbrt-v4 for caustics validation. Not yet evaluated.

### F-HDR-FILM: True float-HDR film buffer + EXR/PFM native output

**Priority:** Medium — unblocks physically-precise validation and post-processing.
The renderer currently tone-maps in-shader and quantizes to 8-bit in the ray payload; there
is no linear HDR film. Sprint 33 adds a quantized-linear PFM dump as a stopgap. A real
float-HDR film buffer (accumulate radiance in `float4`, tone-map as a post-pass, write EXR
via a small encoder or PFM) would remove the 8-bit clamp, decouple validation from the tone
mapper, and enable bloom/exposure post-processing. Requires a ray-payload refactor
(radiance instead of packed bytes).

### F-CAUSTICS-SDS: Caustics visible through/in specular surfaces

Sprint 33 composites caustics only onto primary-ray diffuse hits — no caustic seen *through*
a glass sphere or *in* a mirror (SDS/LSDS paths). Extending the gather to secondary
diffuse hits would close the residual pbrt gap at glass silhouettes and enable
caustic-behind-glass. Ladder rungs C5–C8 would extend to cover it.

### F-MSE-CONVERGENCE: MSE-vs-spp convergence tooling

pbrt's `--mse-reference-image`/`--mse-reference-out` writes MSE vs sample count, quantifying
convergence. An analogous hook on menger's accumulation loop would feed C6-style monitoring
and give an objective "how many samples/photons is enough" answer per scene.

### F-LAYERED-MATERIALS: Layered materials (coateddiffuse / coatedconductor)

pbrt models a dielectric interface layer over a diffuse or conductor base (plastic, varnished
wood, tarnished metal) via Monte-Carlo layer simulation. Would materially improve the
realism of non-metal, non-glass surfaces.

### F-POWER-LIGHTS: Power/illuminance-normalized light specification

pbrt lights accept a `power` (or `illuminance`) parameter and back-solve intensity, giving
unit-correct, resolution-independent lighting. menger uses raw `intensity`. Part of the
historic caustics brightness gap was a light-unit mismatch — power normalization would make
scenes portable and physically meaningful.

### F-LD-SAMPLERS: Low-discrepancy samplers for accumulation

pbrt's default `zsobol` (Owen-scrambled Sobol) converges far faster than independent RNG.
menger's accumulation uses independent sampling. A stratified/low-discrepancy sampler would
cut noise at equal sample count across all rendering (not just caustics).

### F-SPECTRAL-IOR: Measured spectral IOR (Sellmeier) beyond Cauchy

Sprint 32's Cauchy model is a 2-term fit. pbrt supports full spectral `eta` from measured data
(Sellmeier equation, refractiveindex.info). Would improve dispersion accuracy for real
materials (specific glass types, gemstones).

### F-LIVE-PREVIEW: tev/display-server live render preview

pbrt streams to a `tev` display server (`--display-server addr:port`) for live,
progressively-refined preview. Pairs with the real-time-preview backlog item (F17).

### F-PBRT-EXPORT: General DSL → pbrt scene exporter

Sprint 33 hand-authors pbrt twin scenes and adds a triangle-mesh → PLY exporter. A full
`Scene` → `.pbrt` converter (geometry, materials, lights, camera) would let *any* menger
scene be cross-validated against pbrt automatically, not just the hand-picked ladder scenes.

### F-PBR-DIFFUSE: Physically based direct lighting (drop the 0.3 ambient + 0.7 blend)

menger's diffuse shading (`optix-jni helpers.cu:calculateLighting`) is **not physically
based**: it returns `AMBIENT_LIGHT_FACTOR (0.3) + direct · DIFFUSE_BLEND_FACTOR (0.7)` — a
constant 30% ambient fill plus a 70% direct blend. Measured against the analytic
Lambertian direct radiance `ρ/π · I/d² · cosθ`, the constant ambient makes lit floor ~14%
too bright and lifts dark/grazing regions (Sprint 33 diagnosis: menger floor 0.882 vs
pbrt/analytic 0.777 on a matched patch). pbrt is physically correct.

Making menger physically based (ambient → 0, exact `ρ/π`, true `1/d²` instead of the
`1/(1+d²)` softening) would let whole-image comparison against pbrt succeed and remove the
last non-physical "look" knobs. **Blast radius: every committed reference image regenerates**
(all integration + visual tests), so this is its own workstream, not a caustics tweak.
Sprint 33 sidesteps it by validating caustics via the on−off **caustic-delta** metric, which
cancels the constant ambient and the shared direct term. Requires an optix-jni release.
