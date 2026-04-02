# Sprint 20: Materials, Textures & Backgrounds

**Sprint:** 20 - Materials, Textures & Backgrounds
**Status:** Not Started
**Estimate:** ~21 hours
**Branch:** `feature/sprint-20`
**Dependencies:** Sprint 19 (geometry registry, analytical primitives), Sprint 18 (IS programs)

> **MILESTONE: v0.8 — Textures & Backgrounds**

---

## Goal

Add image texture loading, UV generation for all geometry types, environment backgrounds,
procedural textures, and PBR texture maps. Establishes the full material pipeline.

## Success Criteria

- [ ] Image texture loading works (PNG, HDR, EXR via stb_image or similar)
- [ ] UV coordinates generated for all geometry types (parametric, mesh, sphere, sponge)
- [ ] Background images / environment maps / skybox supported
- [ ] Procedural texture infrastructure in shaders (CUDA/OptiX)
- [ ] Wood, marble, and noise procedural textures available
- [ ] Sponge XYZ->RGB procedural texture works
- [ ] PBR normal and roughness maps supported
- [ ] `docs/guide/user-guide.md` materials/textures section updated
- [ ] All tests pass

---

## Tasks

### Task 20.1: Image Texture Loading Infrastructure

**Estimate:** 3h

Add image file loading to the C++ layer and expose through JNI.

#### Implementation

- Add stb_image (or similar) to the C++ build
- Load PNG, HDR, EXR files to host memory
- Upload texture data to CUDA device memory
- Create CUDA texture objects with configurable filtering/wrapping
- JNI methods: `loadTexture(path)`, returning a texture handle

#### optix-jni Boundary

This is general-purpose OptiX infrastructure — add to the optix-jni public API side,
not as Menger-specific convenience.

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

### Task 20.3: Background / Environment Maps / Skybox

**Estimate:** 3h
**Depends on:** 20.1

Replace the current miss shader (single plane or solid color) with image-based backgrounds.

- HDR environment maps for image-based lighting
- Skybox cube maps
- Simple background images (fallback for miss rays)
- Miss shader samples environment map based on ray direction

**Note:** With planes promoted to first-class geometry (Sprint 19.4), the miss shader
is freed to handle only backgrounds.

---

### Task 20.4: Procedural Texture Infrastructure

**Estimate:** 4h

Shader-side infrastructure for procedural textures:
- Texture coordinate generation (UV, triplanar, world-space)
- Noise functions (Perlin, Worley/cellular) in CUDA
- Fractal brownian motion (fBm) compositing
- Mixing/blending utilities

Prerequisite for 20.5 and 20.6.

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

- Activate the existing `normal_texture` and `roughness_texture` fields in
  `MaterialProperties` (currently declared but unused — L-dead-structs)
- Tangent-space normal map sampling in closesthit shaders
- Roughness map sampling affecting specular response
- DSL and CLI syntax for specifying texture file paths per material

---

### Task 20.8: Documentation

**Estimate:** 2h

- Sprint retrospective
- CHANGELOG.md update
- `docs/guide/user-guide.md` materials/textures section
- Example renders showing procedural textures and environment maps
- arc42 update: texture pipeline architecture

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 20.1 | Image texture loading infrastructure | 3h | None |
| 20.2 | UV generation for all geometry | 3h | None |
| 20.3 | Background / environment maps / skybox | 3h | 20.1 |
| 20.4 | Procedural texture infrastructure | 4h | None |
| 20.5 | Wood, marble, noise textures | 2h | 20.4 |
| 20.6 | Sponge XYZ->RGB texture | 1h | 20.4 |
| 20.7 | PBR normal + roughness maps | 3h | 20.1, 20.2 |
| 20.8 | Documentation | 2h | All |
| **Total** | | **~21h** | |

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

### Formerly

This was Sprint 19 in the original roadmap. Renumbered to Sprint 20 because the new
Sprint 18 (GPU Infrastructure) and Sprint 19 (Advanced Geometry) are prerequisites
for the full texture pipeline — UV generation requires the geometry types to exist,
and environment maps benefit from planes being first-class geometry.

### L-dead-structs Resolution

Task 20.7 resolves the `L-dead-structs` code quality issue by activating the
`normal_texture` and `roughness_texture` fields in `MaterialProperties` that were
declared as "(future)" placeholders.
