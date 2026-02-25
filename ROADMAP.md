# Menger Roadmap

**Last Updated:** 2026-02-25

Strategic feature planning for the Menger ray tracing renderer.

---

## Milestones

| Version | Name | Status | Key Features |
|---------|------|--------|--------------|
| v0.4.1 | Full 3D Support | Complete | Materials, textures, multi-object scenes |
| v0.4.2 | 4D Foundation | Complete | Tesseract, cylinder edges, 4D rotation, metallic reflection |
| v0.4.3 | 4D Fractals | Complete | 4D sponges (tesseract-sponge, tesseract-sponge-2), edge rendering |
| v0.5 | Advanced 4D | Complete | Interactive 4D manipulation, LibGDX wrapper, thin-film interference |
| v0.5.1 | Patch Release | Complete | Sprint 11 refinements, CI fixes |
| v0.5.2 | Scene Animation | In Progress | t-parameter animation |
| v0.6 | Visual Quality | Planned | Material enhancements, transparent shadows |

---

## Completed Sprints

| Sprint | Focus | Status | Archive |
|--------|-------|--------|---------|
| 1 | Foundation (ray stats, shadows) | Complete | - |
| 2 | Interactivity (mouse control, lights) | Complete | - |
| 3 | Quality (antialiasing, color API) | Complete | - |
| 4 | Caustics | Deferred | [docs/caustics/](docs/caustics/) |
| 5 | Triangle Mesh + Cube | Complete | [archive](docs/archive/sprints/) |
| 6 | Full Geometry (IAS, sponges) | Complete | [archive](docs/archive/sprints/) |
| 7 | Materials & Textures | Complete | [archive](docs/archive/sprints/) |
| 8 | 4D Projection + UX | Complete | [archive](docs/archive/sprints/) |
| 9 | TesseractSponge & Fractional Levels | Complete | [archive](docs/archive/sprints/) |
| 10 | Scala DSL for Scene Description | Complete | [docs/sprints/](docs/sprints/) |
| 11 | LibGDX Wrapper & Thin-Film Interference | Complete | [docs/sprints/](docs/sprints/) |
| 12 | t-Parameter Animation System | Complete | [docs/sprints/SPRINT12.md](docs/sprints/SPRINT12.md) |

---

## Planned Sprints

### Sprint 13: Visual Quality & Material Enhancements (10-14 hours)

**Goal:** Material realism and visual polish

- Plane materials and textures
- Transparent shadows (colored shadows through glass)
- Material physical correctness validation
- Mixed-metallic material examples (0 < metallic < 1)
- Rounded edges on cubes/sponges (optional stretch goal)

**Note:** This content was originally planned as Sprint 12 but was deferred when Sprint 12 was reprioritized for the t-parameter animation system.

### Sprint 14: Video Output & Visual Enhancements (15-20 hours)

**Goal:** Video output, animation preview, and rendering quality improvements

- Video output via ffmpeg (MP4/WebM from frame sequences)
- Animation preview mode (interactive t-parameter scrubbing)
- Soft shadows with area lights (penumbra)
- Depth of field (camera aperture, bokeh)
- Additional primitives (cylinder, cone, torus)
- Coordinate cross (axis visualization)

**Note:** Replaces the original Sprint 14 ("Advanced Animation System") which was built on the keyframe approach. The t-parameter animation system (Sprint 12) made most keyframe features obsolete.

**MILESTONE: v0.6 - Scene Animation & Visual Quality**

### Sprint 15: Polish & Infrastructure (10-15 hours)

**Goal:** Test coverage, animation workflow, and maintenance

- Bezier/spline camera paths
- Animation export/import (JSON format)
- Test coverage analysis and improvement
- Valgrind standalone test verification

---

## Backlog

Ideas for future consideration, not yet scheduled.

### High Interest

| Idea | Description | Complexity | Status |
|------|-------------|------------|--------|
| Caustics | Progressive Photon Mapping (deferred, algorithm issues) | Very High | Deferred |
| More primitives | Cylinders, cones, torus | Medium | Sprint 14 |
| Soft shadows | Area lights with penumbra | Medium | Sprint 14 |

### Medium Interest

| Idea | Description | Complexity | Status |
|------|-------------|------------|--------|
| Mixed geometry scenes | Spheres + cubes/meshes in same scene (multi-GAS IAS) | Medium | Backlog (TD-5) |
| Depth of field | Camera aperture simulation, bokeh | Medium | Sprint 14 |
| HDR environment | Image-based lighting | Medium | Backlog |
| Subsurface scattering | Advanced material effect | High | Backlog |
| Real-time preview | Interactive low-quality mode | Medium | Backlog |

### Low Priority / Deferred

| Idea | Description | Notes |
|------|-------------|-------|
| Dynamic window resize | LibGDX/OptiX buffer issues | 15+ hours investigation, no resolution |
| GPU composites | Render fractals as composites | Needs design |
| Coordinate cross | Render axis visualization | Sprint 14 |

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
| DSL & 4D UX (10-11) | 2 sprints | ~30 hours |
| t-Parameter Animation (12) | 1 sprint | ~12 hours |
| Visual Quality (13) | 1 sprint | 10-14 hours |
| Video & Enhancements (14) | 1 sprint | 15-20 hours |
| Polish (15) | 1 sprint | 10-15 hours |
| **Total Remaining** | 3 sprints | **35-49 hours** |

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
