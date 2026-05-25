# Image-Based Lighting (IBL) — Design Spec

**Date:** 2026-05-25
**Sprint:** 23
**Tasks:** 23.1, 23.2, 23.3, 23.4

---

## Goal

Extend the renderer so an HDR environment map can illuminate objects, not just serve as a
background. The feature is additive: DSL point/directional lights continue to work; IBL
contributes on top.

Backward compatibility: all existing scenes without `ibl` in their DSL compile and render
identically to Sprint 22 behavior.

---

## Architecture Overview

```
DSL Scene.ibl: Option[IBL]
       │
       ▼
SceneConverter → sets ibl_enabled / ibl_strength / ibl_samples in RenderConfig
       │
       ▼
OptiXWrapper::uploadTextureFromFile()  (existing — handles .hdr extension)
  → stbi_loadf → uploadTextureFloat (existing)
  → computeAndUploadEnvMapCDF()  ← NEW, called before stbi_image_free
       │
       ▼
Params struct (OptiXData.h)
  + ibl_enabled       bool
  + ibl_strength      float
  + ibl_samples       int
  + env_cdf_marginal  cudaTextureObject_t  (1D, height floats)
  + env_cdf_cond      cudaTextureObject_t  (2D, width×height floats)
  + env_pdf           cudaTextureObject_t  (2D, luminance-weighted PDF)
       │
       ▼
helpers.cu :: calculateLighting()   ← NEW IBL sample loop
  → sampleEnvLight()               ← NEW
  → traceShadowRay() (existing)
  → MIS weight applied to both env sample and BSDF reflection ray
```

Existing field `env_map_enabled` / `env_map_texture` in `Params` are unchanged; they
control the miss shader background. `ibl_enabled` is a new, independent boolean.

---

## Data Structures

### `OptiXData.h` additions to `Params`

```cpp
// IBL — image-based lighting (env map as area light)
bool                ibl_enabled;       // false = background only (env_map_enabled path)
float               ibl_strength;      // scales env radiance contribution; default 1.0
int                 ibl_samples;       // env samples per shading point; default 1
cudaTextureObject_t env_cdf_marginal;  // 1D, height floats — marginal CDF over rows
cudaTextureObject_t env_cdf_cond;      // 2D, width×height floats — conditional CDF
cudaTextureObject_t env_pdf;           // 2D, width×height floats — PDF for MIS
```

### `RenderConfig.h` additions

```cpp
bool  ibl_enabled   = false;
float ibl_strength  = 1.0f;
int   ibl_samples   = 1;
```

### Scala DSL additions (`menger-app`)

```scala
// menger/dsl/IBL.scala  (new file)
case class IBL(
  strength: Float = 1.0f,
  samples: Int = 1,
)

// menger/dsl/Scene.scala — add field
case class Scene(
  ...,
  ibl: Option[IBL] = None,   // None = background only; Some(...) = illumination enabled
)
```

---

## Task 23.1 — CDF Precomputation (CPU + GPU upload)

**Where:** `OptiXWrapper.cpp`, new private method `computeEnvMapCDF()`.

**Called from:** `uploadTextureFromFile()`, after `stbi_loadf` succeeds and before
`stbi_image_free`. The raw `float*` pixel data must still be live; CDF is computed from
it on the CPU before the allocation is freed.

**Algorithm:**

1. Luminance per pixel: `L(x,y) = 0.2126f·r + 0.7152f·g + 0.0722f·b` (Rec.709)
2. Row weight: `w(y) = sin(π · (y + 0.5) / height)` (equirectangular solid-angle correction)
3. Weighted luminance: `wL(x,y) = L(x,y) · w(y)`
4. Marginal PDF `p(v)`: row sums of `wL`, then normalize to sum = 1
5. Marginal CDF `F(v)`: prefix-sum of `p(v)`
6. Conditional PDF `p(u|v)`: for each row v, normalize row's `wL` values
7. Conditional CDF `F(u|v)`: prefix-sum of each row's `p(u|v)`
8. PDF texture: `env_pdf(x,y) = L(x,y) · w(y) / total_integral` (unnormalized; MIS uses it
   relative, so normalization constant cancels)

Upload via `cudaMallocArray` / `cudaCreateTextureObject` using the same pattern as
`uploadTextureFloat`, with `cudaChannelFormatKindFloat` (1-channel R32F).

Store texture objects in `m_env_cdf_marginal`, `m_env_cdf_cond`, `m_env_pdf` fields on
`OptiXWrapperImpl`; copy to `Params` at render time alongside the existing
`env_map_texture` copy.

**Memory management:** release in `releaseTextures()` (or a dedicated
`releaseCDFTextures()`) using the existing `cudaDestroyTextureObject` /
`cudaFreeArray` pattern. Only allocate when an HDR env map is loaded.

---

## Task 23.2 — `sampleEnvLight()` Shader Function

**Where:** `helpers.cu`, new `__device__` function.

```cuda
__device__ float3 sampleEnvLight(
    unsigned int& seed,
    float3&       light_dir_out,
    float&        pdf_out
)
```

**Steps:**
1. Sample `v` from `env_cdf_marginal` via binary search (or `tex1D` + linear inversion)
2. Sample `u` from `env_cdf_cond[v]` via binary search
3. Convert `(u, v)` to spherical: `theta = v·π`, `phi = u·2π`
4. Convert to world-space direction
5. Sample `env_map_texture` at `(u, v)` → radiance `L`
6. `pdf_out = tex2D(env_pdf, u, v) · height·width / (2π²)` (solid-angle PDF)
7. Return `L · params.ibl_strength`

**Integration in `calculateLighting()`:** after the existing DSL light loop, add:

```cuda
if (params.ibl_enabled) {
    for (int s = 0; s < params.ibl_samples; ++s) {
        float3 ibl_dir; float ibl_pdf;
        float3 radiance = sampleEnvLight(seed, ibl_dir, ibl_pdf);
        if (ibl_pdf > 0.f && !traceShadowRay(..., ibl_dir)) {
            // accumulate with MIS weight (Task 23.3)
        }
    }
}
```

---

## Task 23.3 — Multiple Importance Sampling (MIS)

**Balance heuristic (β=2):**

```
w_env  = pdf_env²  / (pdf_env²  + pdf_bsdf²)
w_bsdf = pdf_bsdf² / (pdf_env²  + pdf_bsdf²)
```

**In the IBL sample loop (hit shaders + helpers.cu):**
- Evaluate BSDF in `ibl_dir` → `bsdf_val`, compute `pdf_bsdf` (cosine-weighted for
  diffuse, specular lobe for metals/glass)
- Contribution: `radiance · bsdf_val · cos_theta · w_env / ibl_pdf`

**For the existing BSDF reflection ray** (when it hits env map in miss shader):
- Evaluate `env_pdf` at the miss-shader direction → `pdf_env`
- Apply `w_bsdf` weight to the returned env radiance

Starting point: diffuse BSDF only (`pdf_bsdf = cos_theta / π`); specular MIS deferred to
a later sprint if time allows.

---

## Task 23.4 — DSL Wiring

**Scala → C++ path:**

1. `SceneConverter` reads `scene.ibl`; if `Some(IBL(s, n))`, calls
   `renderer.setIBL(strength = s, samples = n)` on the JNI wrapper.
   If `scene.ibl.isDefined` but `scene.envMap.isEmpty`, log a warning and treat as
   `ibl = None` (no IBL without an HDR map).
2. New `@native` declaration: `setIBLNative(strength: Float, samples: Int): Unit` in
   `OptiXRenderer.scala`.
3. C++ JNI impl sets `impl->config.ibl_enabled = true`, `.ibl_strength`, `.ibl_samples`.
4. `Params` copy in `OptiXWrapper.cpp` copies the three RenderConfig fields.
5. CDF textures are copied to `Params` unconditionally when loaded (zero overhead if
   `ibl_enabled = false` since shaders skip the branch).

**CLI:** No new CLI flag in Sprint 23. IBL is DSL-only. `EnvMapDemo` example scene
extended with `ibl = Some(IBL())`.

---

## Testing

**Unit (Scala):**
- `IBL` case class: default values, field validation (samples ∈ [1,8])
- `SceneConverter`: `ibl = None` → `ibl_enabled = false`; `ibl = Some(IBL(0.5f, 2))`
  → `ibl_enabled = true`, correct values forwarded

**C++ (Google Test):**
- `computeEnvMapCDF`: CDF is monotonically non-decreasing, final value = 1.0, all
  values ∈ [0,1]
- CDF for a uniform luminance map is linear

**Integration:**
- `EnvMapDemo` scene: IBL enabled renders without crash, output is non-uniform (fails
  uniform-render check)
- Reference image committed after visual inspection

---

## Risks

| Risk | Mitigation |
|---|---|
| CDF binary search in shader is slow | Start with `ibl_samples=1`; profiling later |
| PDF = 0 for black pixels → division by zero | Guard: `if (pdf > 1e-6f)` before MIS weight |
| CDF texture size (width×height floats) for 4K HDR ~67 MB | Acceptable; release in destructor |
| MIS for specular surfaces is complex | Diffuse MIS only in Sprint 23; specular deferred |

---

## Out of Scope (Sprint 23)

- OptiX AI Denoiser (`optixDenoiserCreate`)
- Specular MIS beyond diffuse approximation
- CLI `--ibl` flag
- Accumulation frames (Task 23.5 is optional)
