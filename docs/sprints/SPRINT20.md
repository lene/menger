# Sprint 20: Materials, Textures & Backgrounds

**Sprint:** 20 - Materials, Textures & Backgrounds
**Status:** In Progress
**Estimate:** ~29 hours
**Branch:** `feature/sprint-20`
**Dependencies:** Sprint 19 (geometry registry, analytical primitives), Sprint 18 (IS programs)

> **MILESTONE: v0.8 — Textures & Backgrounds**

---

## Goal

Add HDR/EXR image texture loading, UV generation for all geometry types, full image-based
lighting (IBL) environment backgrounds, procedural textures, PBR texture maps, and DSL
texture syntax. Also removes two long-standing legacy code paths (miss_plane.cu plane
intersection, CPU 4D projection) and adds triplanar UV mapping and heat map textures.

## Success Criteria

- [x] HDR/EXR texture loading works via stb_image (PNG/JPG already working)
- [ ] UV coordinates generated for all geometry types (parametric, mesh, sphere, sponge)
- [x] Full IBL: environment map contributes to object shading, not just skybox background
- [x] Procedural texture infrastructure in shaders (CUDA/OptiX)
- [x] Wood, marble, and noise procedural textures available
- [x] Sponge XYZ->RGB procedural texture works
- [ ] PBR normal and roughness maps supported
- [ ] DSL syntax for specifying textures per object/material
- [x] Triplanar UV mapping available for sponges and polytopes
- [x] Heat map / color-by-intensity procedural texture available
- [x] Legacy miss_plane.cu plane intersection path removed
- [x] Legacy CPU 4D path (Mesh4D, RotatedProjection) removed
- [ ] `docs/guide/user-guide.md` materials/textures section updated
- [ ] All tests pass

---

## Tasks

### Task 20.1: Image Texture Loading Infrastructure (HDR/EXR)

**Estimate:** 3h

PNG/JPG loading already works via Java ImageIO (`TextureLoader.scala`, `TextureManager.scala`,
`base_color_texture` in `MaterialProperties`). Remaining work: HDR/EXR support for
environment maps (required for full IBL in 20.3).

#### Implementation

- Add stb_image to C++ build (CMakeLists.txt)
- Load HDR (Radiance .hdr) and EXR files to host memory via stb_image
- Upload to CUDA device memory; create CUDA texture objects with float format
- Expose via JNI alongside existing PNG/JPG path

#### optix-jni Boundary

General-purpose OptiX infrastructure — add to the optix-jni public API side.

---

### Task 20.2: UV Generation for All Geometry

**Estimate:** 3h

Add UV coordinate attributes to vertex data for all geometry types.

| Geometry Type | UV Strategy |
|---------------|-------------|
| Parametric surfaces | Natural `(u, v)` from tessellator parameters |
| Spheres | Spherical UV mapping in shader (analytical) |
| Sponges | Position-based UV (world-space XYZ mapped to UV) |
| Polytopes | Position-based UV or face-based UV |
| Planes | World-space planar projection |

#### What Changes

- Add UV buffer to triangle GAS build pipeline
- `ParametricTessellator`: pass `(u, v)` through as vertex attributes
- `Builder` trait: optional UV generation method
- Shaders: read UV from vertex attributes in closesthit programs

---

### Task 20.3: Full IBL Environment Maps / Skybox

**Estimate:** 3h
**Depends on:** 20.1

Replace the current miss shader (solid color fallback) with image-based backgrounds
that also contribute to object lighting (full IBL).

- HDR environment maps sampled by miss shader based on ray direction
- Environment map contributes diffuse and specular lighting to objects
- Skybox cube maps for lower-cost background-only use
- Simple PNG/JPG background images as fallback

**Note:** With planes promoted to first-class geometry (Sprint 19.4), the miss shader
is freed to handle only backgrounds. The legacy plane path in miss_plane.cu is removed
in Task 20.9.

---

### Task 20.4: Procedural Texture Infrastructure

**Estimate:** 4h

Shader-side infrastructure for procedural textures:
- Texture coordinate generation (UV, triplanar, world-space)
- Noise functions (Perlin, Worley/cellular) in CUDA
- Fractal brownian motion (fBm) compositing
- Mixing/blending utilities

Prerequisite for 20.5, 20.6, 20.12, and 20.13.

---

### Task 20.5: Wood, Marble, Noise Procedural Textures

**Estimate:** 2h
**Depends on:** 20.4

Implement wood grain, marble vein, and pure-noise procedural textures using the
infrastructure from 20.4. Available in DSL as material parameters.

---

### Task 20.6: Sponge XYZ->RGB Procedural Texture

**Estimate:** 1h
**Depends on:** 20.4

Map the 3D position of sponge geometry to RGB color (x->R, y->G, z->B or similar),
creating a colorful cross-section visualization.

---

### Task 20.7: PBR Texture Maps (Normal + Roughness)

**Estimate:** 3h
**Depends on:** 20.1, 20.2

Support image-based normal maps and roughness maps for PBR materials.

#### What Changes

- Add `normal_texture` and `roughness_texture` fields to `MaterialProperties` in
  `OptiXData.h` (currently only `base_color_texture` exists; padding[4] has room)
- Tangent-space normal map sampling in closesthit shaders
- Roughness map sampling affecting specular response
- DSL and CLI syntax for specifying texture file paths per material

Resolves the `L-dead-structs` code quality item (the fields were planned but not yet added).

---

### Task 20.8: Documentation

**Estimate:** 2h

- Sprint retrospective
- CHANGELOG.md update
- `docs/guide/user-guide.md` materials/textures section
- Example renders showing procedural textures and environment maps
- arc42 update: texture pipeline architecture

---

### Task 20.9: Remove Legacy miss_plane.cu Plane Intersection Path

**Estimate:** 1h

The miss shader (`miss_plane.cu`) still contains a legacy plane intersection code path
from before planes became first-class geometry in Sprint 19.4. This is dead code marked
for removal. Remove it and simplify the miss shader to handle only backgrounds.

---

### Task 20.10: DSL Texture Syntax

**Estimate:** 2h

Define and implement the user-facing DSL syntax for specifying textures on objects
and materials. Without this, the texture loading and UV infrastructure has no
user-visible interface.

- Syntax design: e.g. `texture = "wood.png"`, `normalMap = "brick_n.png"`
- DSL parser updates in `menger-app/src/main/scala/menger/dsl/`
- CLI equivalent flags for headless mode
- Integration test: scene with texture parameter

---

### Task 20.11: Remove Legacy CPU 4D Path (Mesh4D, RotatedProjection)

**Estimate:** 2h

The GPU projection path is the only 4D path now. The CPU 4D code (`Mesh4D`,
`RotatedProjection`) has been superseded but not yet removed (deferred from TODO.md
across multiple sprints). Remove it.

---

### Task 20.12: Triplanar UV Mapping

**Estimate:** 1.5h
**Depends on:** 20.4

Position-based UV (used for sponges and polytopes in 20.2) produces visible seams and
stretching where geometry faces change direction. Triplanar mapping samples the texture
three times along the XYZ axes and blends by surface normal, eliminating seams.

- Implement in CUDA/OptiX shader as a coordinate mode
- Apply to sponge and polytope geometry types
- Expose as a UV mode option in DSL/material settings

---

### Task 20.13: Heat Map / Color-by-Intensity Procedural Texture

**Estimate:** 1.5h
**Depends on:** 20.4

Map scalar values (ray bounce count, distance from camera, surface normal angle) to
a color gradient. Useful for visualization and debugging, and as an artistic effect.

- Implement gradient lookup in CUDA using fBm infrastructure from 20.4
- Expose selectable scalar source (bounce depth, distance, normal Z)
- DSL: `procedural = "heatmap"` with gradient configuration

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 20.1 | Image texture loading (HDR/EXR via stb_image) | 3h | None |
| 20.2 | UV generation for all geometry | 3h | None |
| 20.3 | Full IBL environment maps / skybox | 3h | 20.1 |
| 20.4 | Procedural texture infrastructure | 4h | None |
| 20.5 | Wood, marble, noise textures | 2h | 20.4 |
| 20.6 | Sponge XYZ->RGB texture | 1h | 20.4 |
| 20.7 | PBR normal + roughness maps | 3h | 20.1, 20.2 |
| 20.8 | Documentation | 2h | All |
| 20.9 | Remove miss_plane.cu legacy plane path | 1h | None |
| 20.10 | DSL texture syntax | 2h | 20.1, 20.2 |
| 20.11 | Remove legacy CPU 4D path | 2h | None |
| 20.12 | Triplanar UV mapping | 1.5h | 20.4 |
| 20.13 | Heat map / color-by-intensity procedural texture | 1.5h | 20.4 |
| **Total** | | **~29h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] `docs/guide/user-guide.md` materials/textures section updated
- [ ] Example renders showing procedural textures and environment maps

---

## Notes

### IBL vs Visual Skybox

Task 20.3 implements full image-based lighting (IBL): the environment map contributes
to object shading, not just the background. This requires HDR/EXR format (float pixels
that can exceed 1.0) — the real luminance ratios in a photo (sun vs. sky vs. shadow)
matter for correct metallic reflections and diffuse lighting.

### Partial Prior Implementation

PNG/JPG texture loading already exists (`TextureLoader.scala`, `TextureManager.scala`,
`base_color_texture` in `MaterialProperties`, `VERTEX_STRIDE_WITH_UV` in `OptiXData.h`).
Task 20.1 focuses on HDR/EXR gap. Task 20.7 adds `normal_texture` and `roughness_texture`
fields (currently absent from `MaterialProperties` — only padding[4] holds their place).

### Formerly

This was Sprint 19 in the original roadmap. Renumbered to Sprint 20 because Sprint 18
(GPU Infrastructure) and Sprint 19 (Advanced Geometry) are prerequisites — UV generation
requires the geometry types to exist, and environment maps benefit from planes being
first-class geometry.
