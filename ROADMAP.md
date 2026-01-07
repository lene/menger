# Menger Roadmap

**Last Updated:** 2026-01-07

Strategic feature planning for the Menger ray tracing renderer.

---

## Milestones

| Version | Name | Status | Key Features |
|---------|------|--------|--------------|
| v0.4.1 | Full 3D Support | ✅ Complete | Materials, textures, multi-object scenes |
| v0.5 | Full 4D Support | Planned | 4D projection, tesseract, 4D sponge |
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

---

## Planned Sprints

### Sprint 8: 4D Projection Foundation (12-18 hours)

**Goal:** Render a tesseract (4D hypercube) projected to 3D

- 4D→3D projection mathematics (perspective, orthographic, cross-section)
- 4D vertex/edge data structures
- Tesseract geometry generation (16 vertices, 32 edges, 24 faces, 8 cells)
- Project tesseract to 3D triangle mesh
- CLI: `--objects type=tesseract`

### Sprint 9: TesseractSponge (15-20 hours)

**Goal:** Render 4D Menger sponge projected to 3D

- TesseractSponge → 4D mesh export
- Apply 4D projection pipeline
- Handle large 4D face counts efficiently
- Progressive level support (levels 0-2+)
- CLI: `--objects type=tesseract-sponge:level=N`

### Sprint 10: 4D Framework (10-15 hours)

**Goal:** Interactive 4D manipulation

- Abstract 4D mesh interface
- 4D rotation/transformation controls
- Interactive 4D manipulation (w-axis rotation)
- CLI: 4D view parameters (`--4d-rotation`, `--4d-slice`)

**🎯 MILESTONE: v0.5 - Full 4D Support**

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

**🎯 MILESTONE: v0.6 - Scene Language & Animation**

---

## Backlog

Ideas for future consideration, not yet scheduled.

### High Interest

| Idea | Description | Complexity |
|------|-------------|------------|
| Caustics | Progressive Photon Mapping (deferred, algorithm issues) | Very High |
| More primitives | Cylinders, cones, torus | Medium |
| Soft shadows | Area lights with penumbra | Medium |

### Medium Interest

| Idea | Description | Complexity |
|------|-------------|------------|
| Depth of field | Camera aperture simulation, bokeh | Medium |
| HDR environment | Image-based lighting | Medium |
| Subsurface scattering | Advanced material effect | High |
| Real-time preview | Interactive low-quality mode | Medium |

### Low Priority / Deferred

| Idea | Description | Notes |
|------|-------------|-------|
| Dynamic window resize | LibGDX/OptiX buffer issues | 15+ hours investigation, no resolution |
| GPU composites | Render fractals as composites | Needs design |
| Coordinate cross | Render axis visualization | Small feature |

---

## Timeline Estimate

| Phase | Sprints | Estimated Hours |
|-------|---------|-----------------|
| Completed (1-7) | 7 sprints | ~100 hours |
| 4D Support (8-10) | 3 sprints | 37-53 hours |
| Scene/Animation (11-13) | 3 sprints | 33-47 hours |
| **Total Remaining** | 6 sprints | **70-100 hours** |

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
