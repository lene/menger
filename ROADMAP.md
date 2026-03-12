# Menger Roadmap

**Last Updated:** 2026-03-12

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
| v0.5.2 | Scene Animation + Visual Quality | In Progress | t-parameter animation, material enhancements, colored shadows |
| v0.6 | Visual Quality + Video | Planned | Video output, soft shadows, depth of field |
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

---

## Planned Sprints

### Sprint 14: Video Output & Visual Enhancements (~19h)

**Goal:** Video output, animation preview, and rendering quality improvements

- **14.1** Video output via ffmpeg (MP4/WebM from frame sequences) (4h)
- **14.2** Animation preview — interactive t scrubbing (3h)
- **14.3** Soft shadows / area lights (3h)
- **14.4** Depth of field / aperture / bokeh (3h)
- **14.5** New primitives: cylinder, cone, torus (3h)
- **14.6** Coordinate cross / axis visualization (1.5h) — depends on 14.5
- **14.7** Documentation (1.5h)
- **14.8** Colored shadows Phase 2: multi-object anyhit accumulation (4–8h, Low priority, TD-6)

**Note:** Replaces the original Sprint 14 ("Advanced Animation System") which was built on the keyframe approach. The t-parameter animation system (Sprint 12) made most keyframe features obsolete.

**MILESTONE: v0.6 — Visual Quality & Video Output**

See [docs/sprints/SPRINT14.md](docs/sprints/SPRINT14.md)

### Sprint 15: DSL Completeness & Infrastructure (~18h)

**Goal:** Finish deferred DSL features and infrastructure/polish work

- **15.1** DSL window/output settings (width, height, saveName, headless) (2h)
- **15.2** Scene composition helpers (SceneBuilder utilities) (2h)
- **15.3** Procedural placement helpers in DSL (2h)
- **15.4** Bezier/spline camera path utility (pure Scala DSL helper) (2h)
- **15.5** Runtime DSL scene evaluation (compile-time → runtime) (2h)
- **15.6** Animation export/import — JSON format for t-param configs (2h)
- **15.7** Optimize pre-push hook (parallelization) (2h)
- **15.8** Better agent instructions + developer docs (1h)
- **15.9** Test coverage improvements + Valgrind CI (1h)

**Note:** Merges deferred Sprint 10 DSL work with infrastructure/polish items.

See [docs/sprints/SPRINT15.md](docs/sprints/SPRINT15.md)

### Sprint 16: Advanced Geometry (~17h)

**Goal:** 4D cutaways, additional polytopes, and parametrized surfaces

- **16.1** 4D and 3D sponge cutaways (clipping plane geometry) (3h)
- **16.2** Other polytopes in 3D (octahedron, dodecahedron, icosahedron) (3h)
- **16.3** Other polytopes in 4D (16-cell, 24-cell, 600-cell) (4h)
- **16.4** Parametrized surfaces in 3D (sphere patches, tori, implicit surfaces) (4h)
- **16.5** User guide: geometry section improvements (1h)
- **16.6** Documentation (2h)

See [docs/sprints/SPRINT16.md](docs/sprints/SPRINT16.md)

### Sprint 17: Textures & Backgrounds (~17h)

**Goal:** Procedural textures, PBR maps, and environment backgrounds

- **17.1** Background images / environment maps / skybox (3h)
- **17.2** Procedural texture infrastructure in shaders (4h)
- **17.3** Wood, marble, noise procedural textures (2h) — depends on 17.2
- **17.4** Sponge XYZ→RGB procedural texture (1h) — depends on 17.2
- **17.5** PBR texture maps: normal + roughness maps (3h)
- **17.6** Test on CUDA 12 and 13 (CI Docker images) (2h)
- **17.7** User guide: materials/textures section improvements (2h)

**MILESTONE: v0.7 — Textures & Backgrounds**

See [docs/sprints/SPRINT17.md](docs/sprints/SPRINT17.md)

### Sprint 18: GPU 4D & Parametrized 4D Surfaces (~14h)

**Goal:** GPU-side 4D math as prerequisite for higher-dimensional geometry

- **18.1** CUDA 4D transform and projection (GPU-side 4D math) (5h)
- **18.2** Parametrized surfaces in 4D (4h) — depends on 16.4 + 18.1
- **18.3** Website with feedback button (GitHub/GitLab issue template) (3h)
- **18.4** Documentation (2h)

See [docs/sprints/SPRINT18.md](docs/sprints/SPRINT18.md)

### Sprint 19: Higher-Dimensional Fractal Analogs (~16h)

**Goal:** Menger and Sierpinski analogs in 4D+

- **19.1** 4D Menger sponge analog (5h) — depends on 18.1
- **19.2** Higher-dimensional Sierpinski tetrahedron analogs (4h) — depends on 18.1
- **19.3** 3D Menger cutaway visualization tools (3h) — depends on 16.1
- **19.4** Exploration / interactive parameter space (2h)
- **19.5** Documentation (2h)

See [docs/sprints/SPRINT19.md](docs/sprints/SPRINT19.md)

---

## Backlog

Ideas for Sprint 20+ consideration.

### Scheduled (Sprints 13-19)

Items in Sprints 13-19 are no longer in the backlog. See Planned Sprints above.

### Long-Term Backlog (Sprint 20+)

| Idea | Description | Complexity |
|------|-------------|------------|
| L-systems in 3D and 4D | Lindenmayer system fractal generation | High |
| Rotopes | Higher-dimensional geometry generation via rotation | Very High |
| Stereoscopic 3D rendering | Left/right eye cameras for VR/3D output | Medium |
| Caustics | Progressive Photon Mapping (deferred, algorithm issues) | Very High |
| Subsurface scattering | Advanced material effect | High |
| Real-time preview | Interactive low-quality mode | Medium |
| HDR environment | Image-based lighting | Medium |

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
| Video & Enhancements (14) | 1 sprint | ~19 hours |
| DSL + Infrastructure (15) | 1 sprint | ~18 hours |
| Advanced Geometry (16) | 1 sprint | ~17 hours |
| Textures & Backgrounds (17) | 1 sprint | ~17 hours |
| GPU 4D (18) | 1 sprint | ~14 hours |
| Higher-Dim Fractals (19) | 1 sprint | ~16 hours |
| **Total Remaining** | 7 sprints | **~112 hours** |

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
