# Sprint 19: Materials, Textures & Backgrounds

**Sprint:** 19 - Materials, Textures & Backgrounds
**Status:** Not Started
**Estimate:** ~15 hours
**Branch:** `feature/sprint-19`
**Dependencies:** Sprint 13 (13.1 — Plane materials, optional), Sprint 18 (18.3 — for PBR maps on surfaces)

> **MILESTONE: v0.7 — Textures & Backgrounds**

---

## Goal

Add environment backgrounds, procedural textures, and PBR texture maps. Establishes
the procedural texture infrastructure that enables wood/marble textures and the sponge
XYZ→RGB effect.

## Success Criteria

- [ ] Background images / environment maps / skybox supported
- [ ] Procedural texture infrastructure in shaders (CUDA/OptiX)
- [ ] Wood, marble, and noise procedural textures available
- [ ] Sponge XYZ→RGB procedural texture works
- [ ] PBR normal and roughness maps supported
- [ ] `docs/guide/user-guide.md` materials/textures section updated
- [ ] All tests pass

---

## Tasks

### Task 19.1: Background Images / Environment Maps / Skybox

**Estimate:** 3h

Support for image-based backgrounds:
- HDR environment maps for image-based lighting
- Skybox cube maps
- Simple background images (fallback for miss rays)

---

### Task 19.2: Procedural Texture Infrastructure

**Estimate:** 4h

Shader-side infrastructure for procedural textures:
- Texture coordinate generation (UV, triplanar, world-space)
- Noise functions (Perlin, Worley/cellular)
- Compositing utilities

This is a prerequisite for 19.3 and 19.4.

---

### Task 19.3: Wood, Marble, Noise Procedural Textures

**Estimate:** 2h

Implement wood grain, marble vein, and pure-noise procedural textures using the
infrastructure from 19.2. Available in DSL as material parameters.

**Depends on:** 19.2

---

### Task 19.4: Sponge XYZ→RGB Procedural Texture

**Estimate:** 1h

Map the 3D position of sponge geometry to RGB color (x→R, y→G, z→B or similar),
creating a colorful cross-section visualization.

**Depends on:** 19.2

---

### Task 19.5: PBR Texture Maps (Normal + Roughness)

**Estimate:** 3h

Support image-based normal maps and roughness maps for PBR materials.
Requires texture sampling infrastructure.

---

### Task 19.6: User Guide — Materials & Textures Section

**Estimate:** 2h

Update `docs/guide/user-guide.md` with:
- Environment map / skybox setup
- Procedural texture usage and parameters
- PBR texture map pipeline

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 19.1 | Background / environment maps / skybox | 3h | None |
| 19.2 | Procedural texture infrastructure | 4h | None |
| 19.3 | Wood, marble, noise textures | 2h | 19.2 |
| 19.4 | Sponge XYZ→RGB texture | 1h | 19.2 |
| 19.5 | PBR normal + roughness maps | 3h | None |
| 19.6 | User guide: materials/textures | 2h | All |
| **Total** | | **~15h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] `docs/guide/user-guide.md` materials/textures section updated
- [ ] Example renders showing procedural textures and environment maps

---

## Dependency Notes

- 19.3 and 19.4 depend on 19.2 (procedural texture infrastructure)
- 19.5 (PBR maps) can be worked on independently
- 19.1 (backgrounds) is independent
- Plane normal/roughness maps (noted in plan as "Sprint 19") follow from 13.1 (plane materials)
- CUDA 12 + 13 CI testing moved to Sprint 16
