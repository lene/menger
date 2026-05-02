# Menger Roadmap

**Last Updated:** 2026-05-02

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
| v0.8 | Advanced Geometry | Planned | 3D/4D polytopes, analytical primitives, geometry registry (Sprint 19) |
| v0.9 | Textures & Materials | Planned | Image textures, procedural textures, PBR maps, environment maps (Sprint 20) |
| v1.0 | Higher-Dimensional Fractals | Planned | 4D Menger/Sierpinski analogs, parameter exploration (Sprint 21) |

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
| 18 | GPU Infrastructure | Complete | [docs/archive/sprints/SPRINT18.md](docs/archive/sprints/SPRINT18.md) |

---

## Planned Sprints

### Sprint 19: Advanced Geometry (~26.5h)

**Goal:** 3D/4D polytopes, analytical primitives, planes as geometry, geometry registry, per-object rotation, render stats, spikes

- **19.1** Additional polytopes in 3D (tetrahedron, octahedron, dodecahedron, icosahedron) (4h)
- **19.2** Additional polytopes in 4D (pentachoron, 16-cell, 24-cell, 120-cell, 600-cell) (6h)
- **19.3** Analytical primitives: cone, torus (2h)
- **19.4** Planes as first-class geometry (2h)
- **19.5** Coordinate cross / axis visualization (1.5h)
- **19.6** Geometry registry (2h)
- **19.7** Per-object 3D rotation via scene graph (3h)
- **19.8** Render time stats per frame and per ray (1h)
- **19.9** Spike: max trace depth above 8 (1h)
- **19.10** Spike: fractional levels with IAS sponges (2h)
- **19.11** Documentation (2h)

See [docs/sprints/SPRINT19.md](docs/sprints/SPRINT19.md)

### Sprint 20: Materials, Textures & Backgrounds (~21h)

**Goal:** Image textures, procedural textures, PBR maps, environment backgrounds

- **20.1** Image texture loading infrastructure (3h)
- **20.2** UV generation for all geometry (3h)
- **20.3** Background / environment maps / skybox (3h)
- **20.4** Procedural texture infrastructure in shaders (4h)
- **20.5** Wood, marble, noise procedural textures (2h)
- **20.6** Sponge XYZ→RGB procedural texture (1h)
- **20.7** PBR texture maps: normal + roughness maps (3h)
- **20.8** Documentation (2h)

**MILESTONE: v0.8 — Textures & Materials**

See [docs/sprints/SPRINT20.md](docs/sprints/SPRINT20.md)

### Sprint 21: Higher-Dimensional Fractals (~14h)

**Goal:** Menger and Sierpinski analogs in 4D+

- **21.1** 4D Menger sponge analog (5h)
- **21.2** Higher-dimensional Sierpinski tetrahedron analogs (4h)
- **21.3** Interactive parameter exploration (1D via scene(t)) (3h)
- **21.4** Documentation (2h)

See [docs/sprints/SPRINT21.md](docs/sprints/SPRINT21.md)

---

## Backlog

Ideas for Sprint 22+ consideration.

### Scheduled (Sprints 13-21)

Items in Sprints 13-21 are no longer in the backlog. See Planned Sprints above.

### Long-Term Backlog (Sprint 22+)

#### Geometry & Rendering (Medium Priority)

| Idea | Description | Complexity | Priority |
|------|-------------|------------|----------|
| Depth of field | Camera aperture simulation, bokeh blur in raygen shader | Medium | Medium |
| Wireframe rendering | Stylistic wireframe via OptiX edge geometry (thin cylinders) | Medium | Medium |
| Fog / depth cue | Distance-based attenuation in shader | Low | Medium |
| Sponge cutaways | 3D/4D sponge cross-sections via clipping planes | Medium | Medium |
| Fractal subdivision on polychora | Use 16-cell, 24-cell, 600-cell as subdivision bases | High | Medium |
| Parametric surface specializations | Spherical coordinates, spherical harmonics | Low | Medium |
| Color by intensity / colormaps | Scalar-to-color mapping for fields and surfaces | Medium | Medium |
| 4D parametrized surfaces | `f(u,v) -> Vec4` on GPU (Clifford torus, hypersphere) | High | Medium |

#### Geometry & Rendering (Low Priority)

| Idea | Description | Complexity | Priority |
|------|-------------|------------|----------|
| L-systems in 3D and 4D | Lindenmayer system fractal generation | High | Low |
| Rotopes | Higher-dimensional geometry generation via rotation | Very High | Low |
| Stereoscopic 3D rendering | Side-by-side/over-under + separate image pairs for VR/3D | Medium | Low |
| Subsurface scattering | Advanced material BSDF effect | High | Low |
| Schläfli polytope generator | Algorithmic construction from `{p,q}` or `{p,q,r}` symbols | Medium (8-12h) | Low |
| Wythoff construction | Full uniform polytopes (Archimedeans, prisms, antiprisms) | Very High (40+h) | Low |

#### Data Visualization

| Idea | Description | Complexity | Priority |
|------|-------------|------------|----------|
| Scalar/vector fields (functions) | GPU evaluation of `f(x,y,z)`, isosurfaces or volume rendering | High | Medium |
| Scalar/vector fields (datasets) | Import VTK/NetCDF files, 3D texture upload | High | Low |
| Multi-dimensional parameter exploration | Independently vary 2-3+ fractal parameters in real-time | Medium | Low |

#### Low Priority / Deferred

| Idea | Description | Notes |
|------|-------------|-------|
| Dynamic window resize | OptiX buffer resize timing issues | 15+ hours investigation, no resolution; LibGDX removed |
| GPU composites | Render fractals as composites | Needs design |

#### Project Infrastructure

| Idea | Description | Complexity |
|------|-------------|------------|
| optix-jni full decoupling | Move Menger-specific methods to adapter layer, generalize Params | Medium (ongoing) |
| optix-jni publication | Full OptiX API wrapper as standalone JVM library | High |
| GitLab Package Registry | Publish artifacts for cross-project dependencies | Low (1-2 hours) |
| Maven Central (optix-jni) | Publish optix-jni as standalone JNI library | Medium (4-6 hours) |
| Repository split | Separate repos for menger-common, optix-jni, menger-app | Medium (depends on above) |
| Remove legacy CPU 4D path | Delete `Mesh4D`, `RotatedProjection` after GPU migration complete | Low |

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
| Advanced Geometry (19) | 1 sprint | ~16.5 hours |
| Materials, Textures & Backgrounds (20) | 1 sprint | ~21 hours |
| Higher-Dimensional Fractals (21) | 1 sprint | ~14 hours |
| **Total Remaining** | 5 sprints | **~110.5 hours** |

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
