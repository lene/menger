# Sprint 23: Image-Based Lighting (IBL)

**Sprint:** 23 - Image-Based Lighting
**Status:** Not Started
**Estimate:** ~22 hours
**Branch:** `feature/sprint-23`
**Dependencies:** Sprint 22 (HDR env map in DSL, tone mapping)

---

## Goal

Make the HDR environment map illuminate scene objects, not just serve as background.
A point on a surface should receive light contributions from all directions in the env map,
weighted by the HDR pixel values. Bright regions in the env map (sun, lamps) cast soft
shadows and contribute specular highlights on nearby objects.

This requires importance-sampled environment lighting in the OptiX pipeline, combined with
Multiple Importance Sampling (MIS) to balance env light sampling against BSDF sampling.

## Success Criteria

- [ ] Objects receive diffuse illumination from HDR environment (no separate DSL lights needed)
- [ ] Specular highlights from bright HDR regions appear on metallic/glass objects
- [ ] IBL integrates with existing DSL lights (additive: DSL lights + env map both contribute)
- [ ] Importance sampling reduces noise vs naive uniform sampling at equal sample count
- [ ] `ibl: Option[IBL] = None` DSL field controls IBL strength and sampling
- [ ] All tests pass

---

## Tasks

### Task 23.1: Environment Map CDF (Importance Sampling Precomputation)

**Estimate:** 4h

To importance-sample the env map, compute a 2D Cumulative Distribution Function (CDF)
on the CPU after HDR load, then upload to GPU.

**Algorithm:**
1. After `stbi_loadf()` in `OptiXWrapper.cpp`, compute per-pixel luminance:
   `L(x,y) = 0.2126*r + 0.7152*g + 0.0722*b` (Rec.709)
2. Account for equirectangular distortion: weight each row by `sin(π * y / height)`
3. Build 2D marginal + conditional CDFs:
   - Row marginal PDF: `p(v)` — sum of luminance per row, normalized
   - Per-row conditional PDF: `p(u|v)` — luminance distribution within each row
4. Upload two 1D/2D textures to GPU:
   - `env_cdf_marginal`: 1D float texture, size = height
   - `env_cdf_conditional`: 2D float texture, size = width × height
5. Add `env_pdf_texture` for MIS weight computation

**GPU data additions in `OptiXData.h`:**
```cpp
cudaTextureObject_t env_cdf_marginal;      // 1D, height floats
cudaTextureObject_t env_cdf_conditional;   // 2D, width × height floats
cudaTextureObject_t env_pdf;               // 2D, luminance-weighted PDF values
```

---

### Task 23.2: Environment Light Sampling in Shader

**Estimate:** 5h
**Depends on:** 23.1

Add `sampleEnvLight()` function in `helpers.cu` that generates a light direction by
importance-sampling the env map CDF, evaluates the env map radiance in that direction,
and computes the PDF for MIS.

**Implementation:**
```cuda
// In helpers.cu — new function
__device__ float3 sampleEnvLight(
    unsigned int& seed,
    float3& light_dir,    // out: sampled direction (world space)
    float& pdf            // out: sampling PDF for MIS
) {
    // 1. Sample v from marginal CDF using binary search or alias method
    // 2. Sample u from conditional CDF[v] using binary search
    // 3. Convert (u,v) → spherical direction → world space ray
    // 4. Evaluate env_map at (u,v) → radiance
    // 5. pdf = luminance(radiance) * sin_theta / (2π² * env_map_integral)
    // 6. Return radiance
}
```

Integrate into main light loop (`helpers.cu:282–353`):
- If `params.ibl_enabled`, call `sampleEnvLight()` for N IBL samples per shading point
- Trace shadow ray in sampled direction; if unoccluded, add radiance contribution

---

### Task 23.3: Multiple Importance Sampling (MIS)

**Estimate:** 5h
**Depends on:** 23.2

Without MIS, env light sampling is noisy for small bright sources (sharp highlights), and
BSDF sampling is noisy for large dim environments. MIS combines both for low variance.

**Balance heuristic MIS weight:**
```
w_env(dir) = pdf_env(dir)² / (pdf_env(dir)² + pdf_bsdf(dir)²)
w_bsdf(dir) = pdf_bsdf(dir)² / (pdf_env(dir)² + pdf_bsdf(dir)²)
```

**Implementation in `hit_triangle.cu` (and other hit shaders):**
1. On each surface hit, draw one env sample (direction + PDF)
2. Evaluate BSDF in that direction → `bsdf_value`, compute `pdf_bsdf`
3. Apply MIS weight: `contribution = radiance * bsdf_value * cos_theta * w_env / pdf_env`
4. Existing reflection ray gets `w_bsdf` weight applied when it hits the env map
5. Accumulate with existing direct light contributions

**Notes:**
- Start with 1 IBL sample per hit; expose `iblSamples: Int = 1` in DSL for quality control
- Power-2 heuristic (`β=2`) is the standard balance; simpler but effective

---

### Task 23.4: DSL Integration

**Estimate:** 3h
**Depends on:** 23.2, 23.3

```scala
case class IBL(
  strength: Float = 1.0f,   // Scale env map contribution (0 = off, 1 = physical)
  samples: Int = 1,          // IBL samples per shading point (1–8)
)

// Scene.scala
case class Scene(
  ...,
  envMap: Option[String] = None,      // Sprint 22
  ibl: Option[IBL] = None,            // Sprint 23 — None = env map background only
  toneMapping: ToneMapping = ToneMapping.Reinhard(),  // Sprint 22
)
```

When `ibl` is `None`, env map is background only (Sprint 22 behavior, backward-compatible).
When `ibl` is `Some(IBL(...))`, env map also illuminates objects.

Wire `strength`, `samples`, `ibl_enabled` into `Params` struct and shaders.

---

### Task 23.5: Noise Reduction / Denoising (Optional, if Time Allows)

**Estimate:** 3h
**Depends on:** 23.2

IBL with 1 sample per hit is inherently noisy. Options:
- **Accumulation frames:** render N frames of the same static scene and average (already
  possible with animation t=const and `frames=N`; needs pixel accumulation buffer)
- **OptiX AI Denoiser:** `optixDenoiserCreate()` — available in OptiX SDK; good quality
  but adds SDK complexity

If time allows, wire up accumulation frames as a `samples: Int` parameter on `RenderSettings`.
Denoiser deferred to later sprint.

---

### Task 23.6: Documentation

**Estimate:** 2h

- User guide: "Image-Based Lighting" section
  - IBL concept: env map as area light
  - DSL: `ibl = Some(IBL(strength = 1.0f, samples = 2))`
  - Interaction with DSL lights (additive)
  - Performance: samples × frames trade-off
- Example renders: metallic sphere, glass object, 4D sponge under HDR illumination
- Sprint retrospective, CHANGELOG.md

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 23.1 | Env map CDF precomputation (CPU + GPU upload) | 4h | Sprint 22 HDR load |
| 23.2 | `sampleEnvLight()` shader function | 5h | 23.1 |
| 23.3 | MIS combining env and BSDF sampling | 5h | 23.2 |
| 23.4 | DSL `IBL` type + `Scene.ibl` wiring | 3h | 23.2, 23.3 |
| 23.5 | Accumulation frames (optional) | 3h | 23.2 |
| 23.6 | Documentation | 2h | All |
| **Total** | | **~22h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] Reference renders: IBL enabled vs disabled comparison committed

---

## Notes

### Why MIS Is Necessary

Without MIS: sampling env map uniformly misses small bright sources (sun = 1 pixel, weight
~0). Sampling BSDF on diffuse surface misses most of the env map. MIS combines both:
env sampling finds the bright regions; BSDF sampling handles the directional response.
Result: lower variance at equal cost.

### Existing Light Loop Compatibility

Current loop in `helpers.cu:282–353` sums contributions from all DSL lights. IBL adds one
more "virtual light" evaluated per hit. Fully additive — existing DSL lights are unaffected.

### IBL and 4D Fractals

4D sponge cross-sections use triangle meshes for hit shading (Sprint 21). The same
`hit_triangle.cu` changes apply. IBL should work for 4D fractal cross-sections without
geometry-specific work.

### Deferred

- **OptiX AI Denoiser** — needs `OptixDenoiser` handle in renderer, separate sprint
- **Prefiltered env map (specular IBL mipmap)** — classic technique for real-time; not
  needed for path-traced offline rendering
