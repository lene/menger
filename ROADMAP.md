# Menger Roadmap

**Last Updated:** 2026-06-10

Strategic feature planning for the Menger ray tracing renderer.
Unscheduled feature ideas: [docs/BACKLOG.md](docs/BACKLOG.md).

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
| v0.7.2 | optix-jni Decoupling | ✅ Complete | optix-jni generic library, menger-geometry layer, MengerRenderer, NativeOptiXApi (Sprints 24-25) |
| v0.7.3 | Repo Split & Code Health | ✅ Complete | menger-common + optix-jni separate repos, Maven Central, all CODE_IMPROVEMENTS resolved (Sprint 26) |
| v0.7.4 | Video Textures & Backgrounds | ✅ Complete | Animated .mp4 video textures + 360° env-map backgrounds via ffmpeg, per-frame GPU texture swap; CUDA 13 / driver ≥580.65 (Sprint 27) |
| v0.7.5 | Agentic Guardrails & Release Automation | ✅ Complete | Cross-repo standards, tiered hooks, multi-model reviews, release-on-merge, stats JSON, benchmarks, Renovate (Sprint 28) |
| v0.7.6 | OptiX API Coverage I | ✅ Complete | AI denoiser, curves primitive, optix-jni release (Sprint 29) |
| TBD | OptiX API Coverage II | ✅ Complete | Architecture hardening — arch review, arc42 audit, API audit, validation mode, SER, 1.0 prep (Sprint 30) |
| TBD | L-Systems | 🔄 In Progress | 3D/4D Lindenmayer systems, turtle geometry, presets (Sprint 31) |
| TBD | Spectral Dispersion | ✅ Complete | Wavelength-dependent IOR, hero-wavelength sampling, diamond fire (Sprint 32) |
| TBD | Production Caustics | 🔄 In Progress | Physically-correct PPM (pbrt-validated), arbitrary geometry, dispersive + reflective caustics (Sprint 33) |
| TBD | PBR Texture Sets | Planned | Shared texture sets (ambientCG/Poly Haven), metallic/AO maps (Sprint 34) |
| TBD | Visual Quality | Pushed back 2026-06-10 | Depth of field, wireframe rendering (Sprint 35) |
| TBD | Data Visualization I | Pushed back 2026-06-10 | Colormaps, scalar fields, isosurfaces, volume rendering (Sprint 36) |
| TBD | 4D Geometry II | Pushed back 2026-06-10 | 4D parametric surfaces, spherical harmonics (Sprint 37) |
| TBD | Advanced Geometry | Pushed back 2026-06-10 | Sponge cutaways, Schläfli generator, polychora fractals (Sprint 38) |

Version numbers for planned milestones are assigned at release time (see release
checklist); they are deliberately not pre-assigned here.

---

## optix-jni Library Releases

Separate versioning track for the standalone `io.github.lene:optix-jni` library.
The library version is independent of the menger app version.

| Version | Status | Description | Sprint |
|---------|--------|-------------|--------|
| 0.1.0 | ⚠️ Defective | Stub library accidentally published — use 0.1.1+ | 26.0 |
| 0.1.1 | ✅ Published | Real CUDA build, Scaladoc, non-GPU tests, Maven Central | 26.0–26.1 |
| 0.1.2 | ✅ Published | Depends on menger-common 0.1.1 (removes dead gpuProject4D field) | 26.13 |
| 1.0.0 | Planned | Stable public API; SemVer stability guarantee | TBD |

**Pre-1.0 contract:** API may change between minor versions.
**1.0 contract:** SemVer stability on `OptiXRenderer`, `NativeOptiXApi`, and all public traits.

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
| 24 | optix-jni Architecture & Foundation | ✅ Complete | [docs/archive/sprints/SPRINT24.md](docs/archive/sprints/SPRINT24.md) |
| 25 | optix-jni Implementation | ✅ Complete | [docs/archive/sprints/SPRINT25.md](docs/archive/sprints/SPRINT25.md) |
| 26 | Repository Split & Code Health | ✅ Complete | [docs/archive/sprints/SPRINT26.md](docs/archive/sprints/SPRINT26.md) |
| 27 | Video Backgrounds | ✅ Complete | [docs/archive/sprints/SPRINT27.md](docs/archive/sprints/SPRINT27.md) |
| 28 | Agentic Guardrails & Release Automation | ✅ Complete | [docs/archive/sprints/SPRINT28.md](docs/archive/sprints/SPRINT28.md) |
| 29 | OptiX API Coverage I | ✅ Complete | [docs/archive/sprints/SPRINT29.md](docs/archive/sprints/SPRINT29.md) |

---

## Planned Sprints

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

### Sprint 28: Agentic Development Guardrails & Release Automation (~45h)

**Goal:** Structural quality enforcement for AI-delivered code: cross-repo standards
with drift detection, tiered hooks with agentic policy checks, hardened local runners,
multi-model (Claude + DeepSeek) MR reviews, release-on-merge with installable-package
proof, enforcement audit

See [docs/sprints/SPRINT28.md](docs/sprints/SPRINT28.md)

---

### Sprint 29: OptiX API Coverage I — Denoiser & Curves (~30h)

**Goal:** OptiX AI denoiser (optix-jni API + DSL/CLI) and curves primitive (swept
tubes); next optix-jni minor release

See [docs/sprints/SPRINT29.md](docs/sprints/SPRINT29.md)

---

### Sprint 30: OptiX API Coverage II — Motion Blur & 1.0 Prep (~26h)

**Goal:** Full OptiX API audit with expose/defer decisions, transform motion blur,
validation mode + SER, optix-jni 1.0 readiness checklist

See [docs/sprints/SPRINT30.md](docs/sprints/SPRINT30.md)

---

### Sprint 31: L-Systems in 3D and 4D (~28h)

**Goal:** Lindenmayer grammar engine, 3D turtle → curve geometry with presets,
DSL/CLI integration, 4D turtle extension

See [docs/sprints/SPRINT31.md](docs/sprints/SPRINT31.md)

---

### Sprint 32: Spectral Dispersion (~22h)

**Goal:** Wavelength-dependent IOR (Cauchy/Abbe model), hero-wavelength sampling,
dispersive material presets — rainbow refraction and diamond fire

See [docs/sprints/SPRINT32.md](docs/sprints/SPRINT32.md)

---

### Sprint 33: Production-Quality Caustics (~77h)

**Goal:** Physically-correct progressive photon mapping, validated against pbrt-v4
(layered validation pyramid: analytic → statistical → converged-reference). Fix the
structural physics defects (emission pdf, Fresnel/reflection, linear compositing, density
estimate), generalize to arbitrary geometry, add dispersive + reflective caustics.

See [docs/sprints/SPRINT33.md](docs/sprints/SPRINT33.md)

---

### Sprint 34: PBR Texture Sets (~24h)

**Goal:** Load shared/downloadable PBR texture sets (ambientCG, Poly Haven) by folder
convention; metallic/AO/height map slots; set metadata (IOR, UV scale)

See [docs/sprints/SPRINT34.md](docs/sprints/SPRINT34.md)

---

### Pushed back 2026-06-10 (planned, after Sprint 34)

| Sprint | Focus | Plan |
|--------|-------|------|
| 35 | Visual Quality — depth of field, wireframe (~20h) | [SPRINT35.md](docs/sprints/SPRINT35.md) |
| 36 | Data Visualization I — colormaps, scalar fields, volumes (~25h) | [SPRINT36.md](docs/sprints/SPRINT36.md) |
| 37 | 4D Geometry II — 4D parametric surfaces, spherical harmonics (~20h) | [SPRINT37.md](docs/sprints/SPRINT37.md) |
| 38 | Advanced Geometry — cutaways, Schläfli generator, polychora fractals (~27h) | [SPRINT38.md](docs/sprints/SPRINT38.md) |

---

## Backlog

Ideas not yet scheduled.

### Evaluated 2026-06-10, not prioritized

Candidates from the 2026-06-10 feature-planning round that were considered and
deliberately deprioritized. Dependencies and tier analysis are in
[docs/sprints/FEATURE_DEPENDENCIES.md](docs/sprints/FEATURE_DEPENDENCIES.md);
detailed per-feature specifications with implementation notes and task-level
estimates are in
[docs/sprints/BACKLOG_FEATURES.md](docs/sprints/BACKLOG_FEATURES.md).

| ID | Idea | Notes |
|----|------|-------|
| F2 | 4D cross-sections (w-slicing) | Low priority per evaluation |
| F3 | Distance-estimator fractals (Mandelbulb, quaternion Julia) | Unblocks F20 |
| F5 | Procedural sun-sky (Hosek-Wilkie) | Feeds existing IBL pipeline |
| F6 | Audio-reactive animation — offline | libav already available (Sprint 27) |
| F7 | Audio-reactive animation — real-time | Needs F17; mic + loopback via PipeWire |
| F8 | Adaptive variance-based sampling | Pairs with the Sprint 29 denoiser |
| F10 | Vector field visualization | Needs scalar-field work (Sprint 36) |
| F11 | Desktop lens window | Needs Sprint 27 in-place texture updates |
| F12a | Stereoscopic still/video rendering | Independent |
| F12b | VR live preview (OpenXR) | Needs F17 + F12a |
| F15 | Volumetric lighting / god rays | Co-design with Sprint 36 ray marching |
| F17 | Real-time progressive preview | Needs denoiser (Sprint 29) — hub for live features |
| F18 | Non-Euclidean / quotient-space rendering | Covers TODO's flat 3-manifolds list |
| F19 | Quasicrystals via cut-and-project | 6D→3D projection of aperiodic lattices |
| F20 | SDF combinators: CSG + fractal morphing | Needs F3 |
| F21 | Gravitational lensing / black hole rendering | Reuses Sprint 36 marching loop |

### Long-Term Backlog

#### Geometry & Rendering (Low Priority)

| Idea | Description | Complexity | Priority |
|------|-------------|------------|----------|
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
| Repository Split & Code Health (26) | 1 sprint | ~28 hours |
| Video Backgrounds (27) | 1 sprint | ~32 hours |
| Agentic Guardrails & Release Automation (28) | 1 sprint | ~45 hours |
| OptiX API Coverage I — Denoiser & Curves (29) | 1 sprint | ~30 hours |
| OptiX API Coverage II — Motion Blur & 1.0 Prep (30) | 1 sprint | ~26 hours |
| L-Systems in 3D and 4D (31) | 1 sprint | ~28 hours |
| Spectral Dispersion (32) | 1 sprint | ~22 hours |
| Production-Quality Caustics (33) | 1 sprint | ~77 hours |
| PBR Texture Sets (34) | 1 sprint | ~24 hours |
| Visual Quality (35, pushed back) | 1 sprint | ~20 hours |
| Data Visualization I (36, pushed back) | 1 sprint | ~25 hours |
| 4D Geometry II (37, pushed back) | 1 sprint | ~20 hours |
| Advanced Geometry (38, pushed back) | 1 sprint | ~27 hours |
| **Total Remaining (27–38)** | 12 sprints | **~323 hours** |

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
