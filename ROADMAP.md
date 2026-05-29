# Menger Roadmap

**Last Updated:** 2026-05-29

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
| v0.5.6 | Developer Infrastructure | Complete | pre-push optimisation, website, CI improvements, AWS spot workflow (Sprint 16) |
| v0.5.7 | Animation & Architecture Foundations | Complete | LibGDX removal, scene graph, engine traits, animation tooling, DSL (Sprint 17) |
| v0.5.8 | GPU Infrastructure | Complete | Multi-GAS IAS, IS programs, GPU 4D math, recursive IAS sponge, maxRayDepth CLI (Sprint 18) |
| v0.6.0 | Advanced Geometry | Complete | Platonic solids, 4-polychora, cone, coordinate cross, geometry registry, 3D rotation, render stats (Sprint 19) |
| v0.6.1 | Textures & Materials | ✅ Complete | Image textures, procedural textures, PBR maps, environment maps (Sprint 20) |
| v0.6.2 | Higher-Dimensional Fractals | ✅ Complete | 4D Menger/Sierpinski analogs, fractional IAS sponge levels, fog, CLI --animate (Sprint 21) |
| v0.7.0 | HDR Environment Maps | ✅ Complete | HDR background, equirectangular env map, tone mapping, Sierpinski4D DSL, JNI safety hardening (Sprint 22) |
| v0.7.1 | Image-Based Lighting | ✅ Complete | IBL with importance sampling + MIS, env map illuminates objects (Sprint 23) |
| v0.8.0 | optix-jni Library | Planned | Generic OptiX JNI library published to Maven Central, menger-geometry layer (Sprints 24-25) |
| v0.8.1 | Repo Split & Code Health | Planned | Separate repos, CUDA leak fixes, legacy code removal (Sprint 26) |
| v0.9.0 | Video Backgrounds | Planned | Animated .mp4 backgrounds via ffmpeg, per-frame GPU texture swap (Sprint 27) |
| v0.9.1 | Visual Quality | Planned | Depth of field, wireframe rendering (Sprint 28) |
| v1.0.0 | Data Visualization I | Planned | Colormaps, scalar fields, isosurfaces, volume rendering (Sprint 29) |

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
| 14 | Rendering Correctness & Code Health | Complete | [docs/sprints/SPRINT14.md](docs/archive/sprints/SPRINT14.md) |
| 15 | Visual Enhancements & Primitives | Complete | [docs/archive/sprints/SPRINT15.md](docs/archive/sprints/SPRINT15.md) |
| 16 | Developer Infrastructure & Website | Complete | [docs/archive/sprints/SPRINT16.md](docs/archive/sprints/SPRINT16.md) |
| 17 | Animation Tooling, DSL & Architecture Foundations | Complete | [docs/archive/sprints/SPRINT17.md](docs/archive/sprints/SPRINT17.md) |
| 19 | Advanced Geometry | Complete | [docs/archive/sprints/SPRINT19.md](docs/archive/sprints/SPRINT19.md) |
| 18 | GPU Infrastructure | Complete | [docs/archive/sprints/SPRINT18.md](docs/archive/sprints/SPRINT18.md) |
| 20 | Materials, Textures & Backgrounds | Complete | [docs/archive/sprints/SPRINT20.md](docs/archive/sprints/SPRINT20.md) |
| 21 | Higher-Dimensional Fractals | ✅ Complete | [docs/archive/sprints/SPRINT21.md](docs/archive/sprints/SPRINT21.md) |
| 22 | HDR Environment Maps | ✅ Complete | [docs/archive/sprints/SPRINT22.md](docs/archive/sprints/SPRINT22.md) |
| 23 | Image-Based Lighting | ✅ Complete | [docs/archive/sprints/SPRINT23.md](docs/archive/sprints/SPRINT23.md) |

---

## Planned Sprints

### Sprint 24: optix-jni Architecture & Foundation (~21h)

**Goal:** Package rename, BaseParams/MengerParams split, menger-geometry scaffold, publication setup

- **24.1** Architecture review + design document (4h)
- **24.2** Package rename `menger.optix` → `io.github.lene.optix` (4h)
- **24.3** Rename `Params` → `BaseParams`, remove 4D/caustics fields (3h)
- **24.4** Create `menger-geometry` sbt subproject skeleton (2h)
- **24.5** Define `MengerParams` struct + validate CUDA compilation (3h)
- **24.6** Publication setup: menger-common (2h)
- **24.7** Publication setup: optix-jni (3h)

See [docs/sprints/SPRINT24.md](docs/sprints/SPRINT24.md)

---

### Sprint 25: optix-jni Implementation (~28h)

**Goal:** Move 4D geometry + caustics to menger-geometry; begin full OptiX API wrapping

- **25.1** Move 4D C++ to menger-geometry (6h)
- **25.2** Move CausticsRenderer to menger-geometry (3h)
- **25.3** Move 4D + caustics Scala/JNI to menger-geometry (4h)
- **25.4** Begin full OptiX API wrapping — core launch path (8h)
- **25.5** Update menger-app dependencies (2h)
- **25.6** Tests + integration validation (3h)
- **25.7** Documentation (2h)

See [docs/sprints/SPRINT25.md](docs/sprints/SPRINT25.md)

---

### Sprint 26: Repository Split & Code Health (~18h)

**Goal:** Separate published artifacts into own repos; fix all open CODE_IMPROVEMENTS items

- **26.1** Repository split (8h)
- **26.2–26.6** Fix 5 medium-priority CODE_IMPROVEMENTS items (8h)
- **26.7** Remove legacy CPU 4D path (2h)

See [docs/sprints/SPRINT26.md](docs/sprints/SPRINT26.md)

---

### Sprint 27: Video Backgrounds (~15h)

**Goal:** Animated `.mp4` video as environment map background, synchronized to animation t

- **27.1** ffmpeg/libav CMake dependency + VideoLoader skeleton (3h)
- **27.2** VideoLoader C++ frame decoder with cache (4h)
- **27.3** JNI binding + Scala VideoLoader (2h)
- **27.4** Per-frame texture swap in animation loop (3h)
- **27.5** DSL `EnvMapVideo` type (1h)
- **27.6** Documentation (2h)

See [docs/sprints/SPRINT27.md](docs/sprints/SPRINT27.md)

---

### Sprint 28: Visual Quality (~20h)

**Goal:** Depth of field (bokeh) and wireframe rendering

- **28.1** Depth of field — lens sampling in raygen (8h)
- **28.2** Wireframe rendering — edge cylinders (6h)
- **28.3** Integration tests + reference images (3h)
- **28.4** Documentation (3h)

See [docs/sprints/SPRINT28.md](docs/sprints/SPRINT28.md)

---

### Sprint 29: Data Visualization I (~25h)

**Goal:** Colormaps, scalar field `f(x,y,z)` GPU evaluation, isosurfaces, volume rendering

- **29.1** Color by intensity / colormaps (5h)
- **29.2** Scalar field GPU evaluation + isosurface (8h)
- **29.3** Volume rendering (ray marching) (8h)
- **29.4** Tests + documentation (4h)

See [docs/sprints/SPRINT29.md](docs/sprints/SPRINT29.md)

---

### Sprint 30: 4D Geometry II (~20h)

**Goal:** 4D parametric surfaces `f(u,v)→Vec4`; spherical harmonics; parametric specializations

- **30.1** 4D parametric surfaces on GPU (8h)
- **30.2** Parametric surface specializations — spherical harmonics (5h)
- **30.3** DSL integration (4h)
- **30.4** Tests + documentation (3h)

See [docs/sprints/SPRINT30.md](docs/sprints/SPRINT30.md)

---

### Sprint 31: Advanced Geometry (~27h)

**Goal:** Sponge cutaways, Schläfli polytope generator, fractal subdivision on polychora

- **31.1** Sponge cutaways via clipping planes (6h)
- **31.2** Schläfli polytope generator (10h)
- **31.3** Fractal subdivision on polychora (8h)
- **31.4** Tests + documentation (3h)

See [docs/sprints/SPRINT31.md](docs/sprints/SPRINT31.md)

---

## Backlog

Ideas not yet scheduled.

### Long-Term Backlog

#### Geometry & Rendering (Low Priority)

| Idea | Description | Complexity | Priority |
|------|-------------|------------|----------|
| L-systems in 3D and 4D | Lindenmayer system fractal generation | High | Low |
| Rotopes | Higher-dimensional geometry generation via rotation | Very High | Low |
| Stereoscopic 3D rendering | Side-by-side/over-under + separate image pairs for VR/3D | Medium | Low |
| Subsurface scattering | Advanced material BSDF effect | High | Low |
| Wythoff construction | Full uniform polytopes (Archimedeans, prisms, antiprisms) | Very High (40+h) | Low |

#### Data Visualization

| Idea | Description | Complexity | Priority |
|------|-------------|------------|----------|
| Scalar/vector fields (datasets) | Import VTK/NetCDF files, 3D texture upload | High | Low |
| Multi-dimensional parameter exploration | Independently vary 2-3+ fractal parameters in real-time | Medium | Low |

#### Low Priority / Deferred

| Idea | Description | Notes |
|------|-------------|-------|
| Dynamic window resize | OptiX buffer resize timing issues | 15+ hours investigation, no resolution; LibGDX removed |
| GPU composites | Render fractals as composites | Needs design |

---

## Timeline Estimate

| Phase | Sprints | Estimated Hours |
|-------|---------|-----------------|
| Completed (1-9) | 9 sprints | ~140 hours |
| DSL & 4D UX (10-11) | 2 sprints | ~30 hours |
| t-Parameter Animation (12) | 1 sprint | ~12 hours |
| Visual Quality (13) | 1 sprint | ~11 hours |
| Rendering Correctness & Code Health (14) | 1 sprint | ~16-25 hours |
| Visual Enhancements & Primitives (15) | 1 sprint | ~13 hours |
| Developer Infrastructure & Website (16) | 1 sprint | ~16 hours |
| Animation, DSL & Architecture (17) | 1 sprint | ~32 hours |
| GPU Infrastructure (18) | 1 sprint | ~27 hours |
| Advanced Geometry (19) | 1 sprint | ~30 hours |
| Materials, Textures & Backgrounds (20) | 1 sprint | ~21 hours |
| Higher-Dimensional Fractals (21) | 1 sprint | ~27 hours |
| HDR Environment Maps (22) | 1 sprint | ~9 hours |
| Image-Based Lighting (23) | 1 sprint | ~42 hours |
| optix-jni Architecture & Foundation (24) | 1 sprint | ~21 hours |
| optix-jni Implementation (25) | 1 sprint | ~28 hours |
| Repository Split & Code Health (26) | 1 sprint | ~18 hours |
| Video Backgrounds (27) | 1 sprint | ~15 hours |
| Visual Quality (28) | 1 sprint | ~20 hours |
| Data Visualization I (29) | 1 sprint | ~25 hours |
| 4D Geometry II (30) | 1 sprint | ~20 hours |
| Advanced Geometry (31) | 1 sprint | ~27 hours |
| **Total Remaining (24–31)** | 8 sprints | **~174 hours** |

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
