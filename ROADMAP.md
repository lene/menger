# Menger Roadmap

**Last Updated:** 2026-04-02

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
| v0.6 | Animation & Architecture Foundations | Planned | LibGDX removal, scene graph, engine traits, animation tooling, DSL (Sprint 17) |
| v0.7 | GPU Infrastructure | Planned | Multi-GAS IAS, IS programs, GPU 4D math (Sprint 18) |
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

---

## Planned Sprints

### Sprint 17: Animation Tooling, DSL & Architecture Foundations (~32h)

**Goal:** Remove LibGDX, introduce scene graph, refactor engines to traits, add animation tooling

- **17.1** Remove LibGDX rendering path (5h)
- **17.2** Engine refactor to trait composition (4h)
- **17.3** Scene graph (transforms + material inheritance) (5h)
- **17.4** All render settings in DSL (4h)
- **17.5** Video output via ffmpeg (MP4/WebM from frame sequences) (4h)
- **17.6** Animation preview — interactive t scrubbing (3h)
- **17.7** Runtime DSL scene evaluation (compile-time → runtime via scala-compiler) (2h)
- **17.8** Bezier/spline camera path (pure Scala DSL helper) (2h)
- **17.9** Procedural placement helpers in DSL (2h)
- **17.10** Define optix-jni API boundary (design only) (1h)

**MILESTONE: v0.6 — Animation & Architecture Foundations**

See [docs/sprints/SPRINT17.md](docs/archive/sprints/SPRINT17.md)

### Sprint 18: GPU Infrastructure (~27h)

**Goal:** Multi-GAS IAS, intersection program infrastructure, GPU-side 4D math,
plus a recursive-IAS sponge demo, `maxRayDepth` CLI completion, and a
failed-render diagnostic.

- **18.1** Multi-GAS Instance Acceleration Structure / TD-5 (8h)
- **18.2** Intersection program infrastructure (4h)
- **18.3** GPU 4D transform and projection (5h)
- **18.4** Recursive IAS Menger sponge (4h)
- **18.5** `maxRayDepth` CLI + verification (2h)
- **18.6** Failed-render diagnostic (1h)
- **18.7** Documentation (3h)

See [docs/sprints/SPRINT18.md](docs/sprints/SPRINT18.md)

### Sprint 19: Advanced Geometry (~16.5h)

**Goal:** 3D/4D polytopes, analytical primitives, planes as geometry, geometry registry

- **19.1** Additional polytopes in 3D (octahedron, dodecahedron, icosahedron) (3h)
- **19.2** Additional polytopes in 4D (16-cell, 24-cell, 600-cell) (4h)
- **19.3** Analytical primitives: cone, torus (2h)
- **19.4** Planes as first-class geometry (2h)
- **19.5** Coordinate cross / axis visualization (1.5h)
- **19.6** Geometry registry (2h)
- **19.7** Documentation (2h)

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
