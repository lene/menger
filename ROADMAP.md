# Menger Roadmap

**Last Updated:** 2026-01-26

Strategic feature planning for the Menger ray tracing renderer.

---

## Milestones

| Version | Name | Status | Key Features |
|---------|------|--------|--------------|
| v0.4.1 | Full 3D Support | ✅ Complete | Materials, textures, multi-object scenes |
| v0.4.2 | 4D Foundation | ✅ Complete | Tesseract, cylinder edges, 4D rotation, metallic reflection |
| v0.4.3 | 4D Fractals | ✅ Complete | 4D sponges (tesseract-sponge, tesseract-sponge-2), edge rendering |
| v0.5 | Advanced 4D | Planned | Interactive 4D manipulation, 4D slicing |
| v0.6 | Scene Language | Planned | Declarative scene files, animation |

---

## Completed Sprints

| Sprint | Focus | Status | Archive |
|--------|-------|--------|---------|
| 1 | Foundation (ray stats, shadows) | ✅ Complete | - |
| 2 | Interactivity (mouse control, lights) | ✅ Complete | - |
| 3 | Quality (antialiasing, color API) | ✅ Complete | - |
| 4 | Caustics | ⏸️ Deferred | [docs/caustics/](docs/caustics/) |
| 5 | Triangle Mesh + Cube | ✅ Complete | [archive](docs/archive/sprints/) |
| 6 | Full Geometry (IAS, sponges) | ✅ Complete | [archive](docs/archive/sprints/) |
| 7 | Materials & Textures | ✅ Complete | [archive](docs/archive/sprints/) |
| 8 | 4D Projection + UX | ✅ Complete | [archive](docs/archive/sprints/) |
| 9 | TesseractSponge & Fractional Levels | ✅ Complete | [archive](docs/archive/sprints/) |

---

## Planned Sprints

### Sprint 10: Scala DSL for Scene Description (18-23 hours)

**Goal:** Type-safe scene definition DSL

- Scala-based DSL that compiles with project
- Block-style and case-class syntax
- Concise object/material/light definitions
- Scene files can import other files
- CLI: `--scene scenes.MyScene`

**Note:** Prioritized over 4D features for better scene authoring workflow

### Sprint 11: 4D Framework Enhancements (8-10 hours)

**Goal:** Complete 4D manipulation UX

- Shift+Scroll for 4D projection adjustment
- ESC to reset 4D view
- CLI shortcuts: `--4d-rotation`, `--4d-preset`
- State persistence (save/load 4D view)
- Per-instance 4D parameters (optional)

**Note:** Core 4D features already complete (Sprints 8-9). This sprint adds convenience features.

**🎯 MILESTONE: v0.5 - Advanced 4D**

### Sprint 12: Visual Quality & Materials (10-14 hours)

**Goal:** Material realism and visual polish

- Plane materials and textures
- Transparent shadows (colored shadows through glass)
- Material physical correctness validation
- Mixed-metallic material examples (0 < metallic < 1)
- Rounded edges on cubes/sponges (optional)

### Sprint 13: Object Animation Foundation (12-18 hours)

**Goal:** Animated scene rendering

- Keyframe-based animation system
- Object transform interpolation (position, rotation, scale)
- Frame sequence rendering
- Output to image sequence (PNG)
- CLI: `--animate-scene`, `--frames`, `--fps`
- DSL animation syntax (if Sprint 10 complete)

### Sprint 14: Advanced Animation (21 hours)

**Goal:** Rich animation capabilities

- Easing functions (linear, ease-in-out, cubic, bounce, elastic)
- Generic property animation (colors, IOR, camera, lights)
- Camera animation (path following)
- Light animation
- Video output via ffmpeg
- Animation preview mode

### Sprint 15: Visual Enhancements & Polish (20-25 hours)

**Goal:** Improve visual quality and animation workflow

#### Visual Quality (10-12h)
- Soft shadows with area lights (penumbra)
- Depth of field (camera aperture, bokeh)
- Coordinate cross (axis visualization for debugging)

#### Animation Workflow (8-10h)
- Bezier/spline camera paths
- Animation export/import (JSON format)
- Additional primitives for animation demos (cylinder, cone, torus)

#### Maintenance (2-3h)
- Test coverage analysis and improvement
- Valgrind standalone test verification

**🎯 MILESTONE: v0.6 - Scene Language & Animation**

**Note:** Sprint 15 continues the original Sprint 14 plan.

---

## Backlog

Ideas for future consideration, not yet scheduled.

### High Interest

| Idea | Description | Complexity | Status |
|------|-------------|------------|--------|
| Caustics | Progressive Photon Mapping (deferred, algorithm issues) | Very High | Deferred |
| More primitives | Cylinders, cones, torus | Medium | → Sprint 14 |
| Soft shadows | Area lights with penumbra | Medium | → Sprint 14 |

### Medium Interest

| Idea | Description | Complexity | Status |
|------|-------------|------------|--------|
| Mixed geometry scenes | Spheres + cubes/meshes in same scene (multi-GAS IAS) | Medium | Backlog (TD-5) |
| Depth of field | Camera aperture simulation, bokeh | Medium | → Sprint 14 |
| HDR environment | Image-based lighting | Medium | Backlog |
| Subsurface scattering | Advanced material effect | High | Backlog |
| Real-time preview | Interactive low-quality mode | Medium | Backlog |

### Low Priority / Deferred

| Idea | Description | Notes |
|------|-------------|-------|
| Dynamic window resize | LibGDX/OptiX buffer issues | 15+ hours investigation, no resolution |
| GPU composites | Render fractals as composites | Needs design |
| Coordinate cross | Render axis visualization | → Sprint 14 |

### Project Infrastructure

| Idea | Description | Complexity |
|------|-------------|------------|
| GitLab Package Registry | Publish artifacts for cross-project dependencies | Low (1-2 hours) |
| Maven Central (optix-jni) | Publish optix-jni as standalone JNI library | Medium (4-6 hours) |
| Repository split | Separate repos for menger-common, optix-jni, menger-app | Medium (depends on above) |

---

## Timeline Estimate

| Phase | Sprints | Estimated Hours |
|-------|---------|-----------------|
| Completed (1-9) | 9 sprints | ~140 hours |
| DSL & 4D UX (10-11) | 2 sprints | 26-33 hours |
| Visual Quality (12) | 1 sprint | 10-14 hours |
| Animation (13-14) | 2 sprints | 33-39 hours |
| Polish (15) | 1 sprint | 20-25 hours |
| **Total Remaining** | 6 sprints | **89-111 hours** |

---

## References

### Ray Tracing
- "Physically Based Rendering" (PBRT) by Pharr, Jakob, Humphreys
- "Ray Tracing in One Weekend" series by Peter Shirley

### 4D Geometry
- "Regular Polytopes" by H.S.M. Coxeter
- [4D Visualization](https://eusebeia.dyndns.org/4d/vis/vis) - Eusebeia

### OptiX
- [NVIDIA OptiX Programming Guide](https://raytracing-docs.nvidia.com/optix7/guide/index.html)
- [OptiX API Reference](https://raytracing-docs.nvidia.com/optix7/api/index.html)
