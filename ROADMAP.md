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
| 8 | 4D Projection + UX | ✅ Complete | [SPRINT8.md](docs/sprints/SPRINT8.md) |
| 9 | TesseractSponge | ✅ Complete | [SPRINT9.md](docs/sprints/SPRINT9.md) |

---

## Planned Sprints

### Sprint 10: 4D Framework (10-15 hours)

**Goal:** Interactive 4D manipulation

- Abstract 4D mesh interface
- 4D rotation/transformation controls
- Interactive 4D manipulation (w-axis rotation)
- CLI: 4D view parameters (`--4d-rotation`, `--4d-slice`)

**🎯 MILESTONE: v0.5 - Advanced 4D**

### Sprint 11: Scene Description Language (15-20 hours)

**Goal:** Declarative scene files

- Design scene file format (YAML or custom DSL)
- Parse scene files with object definitions
- Material and light definitions in scene file
- Per-object transforms in scene file
- CLI: `--scene <file>`

### Sprint 12: Object Animation Foundation (10-15 hours)

**Goal:** Animated scene rendering

- Animation timeline/keyframe data structure
- Object transform interpolation
- Frame sequence rendering
- Output to image sequence (PNG)
- CLI: `--animate`, `--frames`, `--fps`

### Sprint 13: Animation Enhancements (8-12 hours)

**Goal:** Rich animation capabilities

- Easing functions (linear, ease-in-out, cubic, etc.)
- Multi-object animation
- Camera animation (path following)
- Animation preview mode

### Sprint 14: Visual Enhancements & Polish (20-25 hours)

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
| Completed (1-8) | 8 sprints | ~120 hours |
| 4D Support (9-10) | 2 sprints | 25-35 hours |
| Scene/Animation (11-13) | 3 sprints | 33-47 hours |
| Polish (14) | 1 sprint | 20-25 hours |
| **Total Remaining** | 6 sprints | **78-107 hours** |

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
