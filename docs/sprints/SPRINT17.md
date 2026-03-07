# Sprint 17: Textures & Backgrounds

**Sprint:** 17 - Textures & Backgrounds
**Status:** Not Started
**Estimate:** ~17 hours
**Branch:** `feature/sprint-17`
**Dependencies:** Sprint 13 (13.1 — Plane materials, optional), Sprint 16 (16.4 — for PBR maps on surfaces)

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
- [ ] Tested on both CUDA 12 and CUDA 13 via CI Docker images
- [ ] USER_GUIDE.md materials/textures section updated
- [ ] All tests pass

---

## Tasks

### Task 17.1: Background Images / Environment Maps / Skybox

**Estimate:** 3h

Support for image-based backgrounds:
- HDR environment maps for image-based lighting
- Skybox cube maps
- Simple background images (fallback for miss rays)

---

### Task 17.2: Procedural Texture Infrastructure

**Estimate:** 4h

Shader-side infrastructure for procedural textures:
- Texture coordinate generation (UV, triplanar, world-space)
- Noise functions (Perlin, Worley/cellular)
- Compositing utilities

This is a prerequisite for 17.3 and 17.4.

---

### Task 17.3: Wood, Marble, Noise Procedural Textures

**Estimate:** 2h

Implement wood grain, marble vein, and pure-noise procedural textures using the
infrastructure from 17.2. Available in DSL as material parameters.

**Depends on:** 17.2

---

### Task 17.4: Sponge XYZ→RGB Procedural Texture

**Estimate:** 1h

Map the 3D position of sponge geometry to RGB color (x→R, y→G, z→B or similar),
creating a colorful cross-section visualization.

**Depends on:** 17.2

---

### Task 17.5: PBR Texture Maps (Normal + Roughness)

**Estimate:** 3h

Support image-based normal maps and roughness maps for PBR materials.
Requires texture sampling infrastructure.

---

### Task 17.6: Test on CUDA 12 and 13

**Estimate:** 2h

Add CI Docker images for CUDA 12 and CUDA 13, run the full test suite on both.
Document any version-specific workarounds.

---

### Task 17.7: User Guide — Materials & Textures Section

**Estimate:** 2h

Update USER_GUIDE.md with:
- Environment map / skybox setup
- Procedural texture usage and parameters
- PBR texture map pipeline

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 17.1 | Background / environment maps / skybox | 3h | None |
| 17.2 | Procedural texture infrastructure | 4h | None |
| 17.3 | Wood, marble, noise textures | 2h | 17.2 |
| 17.4 | Sponge XYZ→RGB texture | 1h | 17.2 |
| 17.5 | PBR normal + roughness maps | 3h | None |
| 17.6 | Test on CUDA 12 + 13 | 2h | None |
| 17.7 | User guide: materials/textures | 2h | All |
| **Total** | | **~17h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing on CUDA 12 and CUDA 13
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] USER_GUIDE.md materials/textures section updated
- [ ] Example renders showing procedural textures and environment maps

---

## Dependency Notes

- 17.3 and 17.4 depend on 17.2 (procedural texture infrastructure)
- 17.5 (PBR maps) can be worked on independently
- 17.1 (backgrounds) is independent
- Plane normal/roughness maps (noted in plan as "Sprint 17") follow from 13.1 (plane materials)
