# Menger Roadmap

**Last Updated:** 2026-03-28

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
| v0.5.2 | t-Parameter Animation | Complete | t-parameter animation system |
| v0.5.3 | Visual Quality | Complete | material enhancements, colored shadows |
| v0.5.4 | Rendering Correctness & Code Health | Complete | shadow fixes, caustic improvements, code health (Sprint 14) |
| v0.5.5 | Visual Enhancements | Complete | area lights, soft shadows, parametric surfaces, caustics (Sprint 15) |
| v0.6 | Full 3D Rendering | Planned | video output, full DSL/CLI control, animation tooling (Sprints 16–17) |
| v0.7 | Textures & Backgrounds | Planned | Procedural textures, PBR maps, environment maps |

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
| 13 | Visual Quality & Material Enhancements | Complete | [docs/sprints/SPRINT13.md](docs/sprints/SPRINT13.md) |
| 14 | Rendering Correctness & Code Health | Complete | [docs/sprints/SPRINT14.md](docs/sprints/SPRINT14.md) |
| 15 | Visual Enhancements & Primitives | Complete | [docs/archive/sprints/SPRINT15.md](docs/archive/sprints/SPRINT15.md) |

---

## Planned Sprints

### Sprint 16: Developer Infrastructure & Website (~16h)

> ⚠️ **Foundation brainstorm at sprint start:** Before Sprint 18 (higher-dimensional polytopes),
> run a brainstorming session to identify any missing foundations: rendering quality, developer
> tools, DSL/CLI control, materials, textures, backgrounds, etc. Any gaps found here can be
> scheduled into Sprints 16–17 or as new sprints before 18, while there is still time to plan.

**Goal:** Pre-push hook optimisation, developer docs, CI improvements, website, CUDA compatibility, AWS spot rendering

- **16.1** Optimise pre-push hook (parallelization) (2h)
- **16.2** Developer documentation & AGENTS.md refresh (2h)
- **16.3** Test coverage improvements + Valgrind CI (2h)
- **16.4** Project website with GitHub/GitLab feedback button (3h)
- **16.5** Test on CUDA 12 and 13 (CI Docker images) (2h)
- **16.6** AWS spot instance rendering (spike + implementation) (~5h, open-ended)

See [docs/sprints/SPRINT16.md](docs/sprints/SPRINT16.md)

### Sprint 17: Animation Tooling & DSL (~19h)

**Goal:** Video output, animation preview, and DSL convenience helpers

- **17.1** Video output via ffmpeg (MP4/WebM from frame sequences) (4h)
- **17.2** Animation preview — interactive t scrubbing (3h)
- **17.3** DSL window/output settings (width, height, saveName, headless) (2h)
- **17.4** Scene composition helpers (SceneBuilder utilities) (2h)
- **17.5** Procedural placement helpers in DSL (2h)
- **17.6** Bezier/spline camera path (pure Scala DSL helper) (2h)
- **17.7** Runtime DSL scene evaluation (compile-time → runtime) (2h)
- **17.8** Animation export/import — JSON format for t-param configs (2h)

**MILESTONE: v0.6 — Full 3D Rendering**

See [docs/sprints/SPRINT17.md](docs/sprints/SPRINT17.md)

### Sprint 18: Advanced Geometry (~13.5h)

**Goal:** Additional polytopes in 3D and 4D, DSL primitives (cone, torus), coordinate cross

- **18.1** Additional polytopes in 3D (octahedron, dodecahedron, icosahedron) (3h)
- **18.2** Additional polytopes in 4D (16-cell, 24-cell, 600-cell) (4h)
- **18.3** DSL primitives: cone, torus (cylinder TBC) — backed by Sprint 15.2 parametrics (2h)
- **18.4** Coordinate cross / axis visualization (1.5h)
- **18.5** User guide: geometry section improvements (1h)
- **18.6** Documentation (2h)

Note: Sponge cutaways moved to backlog.

See [docs/sprints/SPRINT18.md](docs/sprints/SPRINT18.md)

### Sprint 19: Materials, Textures & Backgrounds (~15h)

**Goal:** Procedural textures, PBR maps, and environment backgrounds

- **19.1** Background images / environment maps / skybox (3h)
- **19.2** Procedural texture infrastructure in shaders (4h)
- **19.3** Wood, marble, noise procedural textures (2h) — depends on 19.2
- **19.4** Sponge XYZ→RGB procedural texture (1h) — depends on 19.2
- **19.5** PBR texture maps: normal + roughness maps (3h)
- **19.6** User guide: materials/textures section improvements (2h)

**MILESTONE: v0.7 — Textures & Backgrounds**

See [docs/sprints/SPRINT19.md](docs/sprints/SPRINT19.md)

### Sprint 20: GPU 4D Infrastructure (~11h)

**Goal:** GPU-side 4D math as prerequisite for higher-dimensional geometry

- **20.1** CUDA 4D transform and projection (GPU-side 4D math) (5h)
- **20.2** Parametrized surfaces in 4D (4h) — depends on 15.2 + 20.1
- **20.3** Documentation (2h)

See [docs/sprints/SPRINT20.md](docs/sprints/SPRINT20.md)

### Sprint 21: Higher-Dimensional Fractals (~16h)

**Goal:** Menger and Sierpinski analogs in 4D+

- **21.1** 4D Menger sponge analog (5h) — depends on 20.1
- **21.2** Higher-dimensional Sierpinski tetrahedron analogs (4h) — depends on 20.1
- **21.3** Exploration / interactive parameter space (3h)
- **21.4** Documentation (2h)

See [docs/sprints/SPRINT21.md](docs/sprints/SPRINT21.md)

---

## Backlog

Ideas for Sprint 22+ consideration.

### Scheduled (Sprints 13-21)

Items in Sprints 13-21 are no longer in the backlog. See Planned Sprints above.

### Long-Term Backlog (Sprint 22+)

| Idea | Description | Complexity |
|------|-------------|------------|
| L-systems in 3D and 4D | Lindenmayer system fractal generation | High |
| Rotopes | Higher-dimensional geometry generation via rotation | Very High |
| Stereoscopic 3D rendering | Left/right eye cameras for VR/3D output | Medium |
| Depth of field | Camera aperture simulation, bokeh blur | Medium |
| Subsurface scattering | Advanced material effect | High |
| Real-time preview | Interactive low-quality mode | Medium |
| HDR environment | Image-based lighting | Medium |
| Sponge cutaways | 4D and 3D sponge cross-sections via clipping planes | Medium |
| Analytical ray intersection | Custom OptiX intersection programs for cylinder, cone, torus (exact, no tessellation) | Medium |

### Low Priority / Deferred

| Idea | Description | Notes |
|------|-------------|-------|
| Dynamic window resize | LibGDX/OptiX buffer issues | 15+ hours investigation, no resolution |
| GPU composites | Render fractals as composites | Needs design |
| Mixed geometry scenes | Spheres + cubes/meshes in same scene (multi-GAS IAS) | Backlog (TD-5) |

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
| Visual Quality (13) | 1 sprint | ~11 hours |
| Rendering Correctness & Code Health (14) | 1 sprint | ~16–25 hours |
| Visual Enhancements & Primitives (15) | 1 sprint | ~13 hours |
| Developer Infrastructure & Website (16) | 1 sprint | ~16 hours |
| Animation Tooling & DSL (17) | 1 sprint | ~19 hours |
| Advanced Geometry (18) | 1 sprint | ~15.5 hours |
| Materials, Textures & Backgrounds (19) | 1 sprint | ~15 hours |
| GPU 4D Infrastructure (20) | 1 sprint | ~11 hours |
| Higher-Dimensional Fractals (21) | 1 sprint | ~16 hours |
| **Total Remaining** | 8 sprints | **~116–126 hours** |

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
