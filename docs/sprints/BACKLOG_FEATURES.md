# Backlog Feature Specifications

**Last Updated:** 2026-06-10
**Status:** Companion to [FEATURE_DEPENDENCIES.md](FEATURE_DEPENDENCIES.md). Detailed
descriptions, implementation notes, and effort estimates for every feature that was
evaluated on 2026-06-10 but **not** scheduled into Sprints 28–38. When one of these is
prioritized, its section here is the seed of the sprint plan.

Feature IDs match FEATURE_DEPENDENCIES.md. Dependencies use the renumbered sprint
scheme (Sprints 28–38, see FEATURE_DEPENDENCIES.md §0).

## Effort Overview

| ID | Feature | Estimate | Hard dependencies |
|----|---------|----------|-------------------|
| F2 | 4D cross-sections (w-slicing) | ~16h | — |
| F3 | Distance-estimator fractals | ~30h | — |
| F5 | Procedural sun-sky (Hosek-Wilkie) | ~16h | — |
| F6 | Audio-reactive animation — offline | ~24h | Sprint 27 (libav) |
| F7 | Audio-reactive animation — real-time | ~26h | F17, F6 |
| F8 | Adaptive variance-based sampling | ~18h | — (pairs with Sprint 29 denoiser) |
| F10 | Vector field visualization | ~26h | Sprint 36 (36.1/36.2), Sprint 29 (curves) |
| F11 | Desktop lens window | ~20h | Sprint 27 (in-place texture update) |
| F12a | Stereoscopic still/video rendering | ~14h | — |
| F12b | VR live preview (OpenXR) | ~40h | F17, F12a |
| F15 | Volumetric lighting / god rays | ~24h | soft: Sprint 36 (36.3), F5 |
| F17 | Real-time progressive preview | ~26h | Sprint 29 (denoiser) |
| F18 | Non-Euclidean / quotient-space rendering | ~40h | — |
| F19 | Quasicrystals via cut-and-project | ~22h | — |
| F20 | SDF combinators: CSG + morphing | ~16h | F3 |
| F21 | Gravitational lensing / black hole | ~30h | soft: Sprint 36 (36.3) |
| B-WYTHOFF | Wythoff construction (uniform polytopes) | ~45h | Sprint 38 (38.2) |
| B-ROTOPES | Rotopes | ~30h + spike | soft: Sprint 37 |
| B-SSS | Subsurface scattering | ~28h | soft: Sprint 29 (denoiser) |
| B-DATASETS | VTK/NetCDF dataset import | ~30h | Sprint 36 |
| B-PARAMEXP | Multi-dimensional parameter exploration | ~20h | F17 |
| T-MTLX | MaterialX support (Layers 1–3) | ~70h | Sprint 33 (map slots) |
| T-SOR | Surface of rotation | ~10h | Sprint 37 (37.1/37.2) |
| T-STAR | Regular star 4-polytopes | ~14h | Sprint 38 (38.2) |
| T-SEMI | Semiregular polyhedra/polytopes | ~16h | Sprint 38 (38.2); superseded by B-WYTHOFF |
| T-CFUNC | Functions ℂ→ℂ | ~12h | Sprint 36 or 37 |
| T-HOPF | Hopf fibration / 3-sphere | ~16h | Sprint 29 (curves) |
| T-SPLAT | Gaussian splats | ~50h + spike | — |
| T-TRACE | 4D spacetime trace | ~30h + spike | — |
| T-360MOVIE | Level-sweep movie with 360° background | ~8h | Sprint 27 |
| T-SCALE | Per-axis scaling | ~6h | — |
| T-SHEAR | Shearing transforms | ~6h | — |

Total if everything were built: ~740h. These are independent options, not a plan.

---

## F2: 4D Cross-Sections (w-Slicing) — ~16h

Slice 4D objects with the hyperplane `w = w₀` instead of projecting them — the second
classic 4D visualization. Animating `w₀` through a tesseract sponge shows level-set
"growth and decay" that projection cannot.

**Implementation notes:**
- Mesh path (polychora, tesseract sponges): intersect each 4D cell/face with the
  hyperplane. For the quad-based 4D meshes this is edge–hyperplane interpolation per
  quad: classify vertices by sign of `w − w₀`, emit the intersection polygon
  (0–4 vertices per quad), triangulate. Output is a closed 3D mesh fed through the
  existing `setTriangleMesh` path
- IFS path (`menger4d`, `sierpinski4d`, `hexadecachoron4d`): the iterative IS shader
  already evaluates membership in 4D; a slice mode pins `w = w₀` in the iteration
  instead of using the projected ray — cheaper than the projection path
- `slice-w` parameter mutually exclusive with projection parameters (`eye-w`,
  `screen-w`); animatable via `--animate slice-w=-1..1`
- Empty-slice frames (hyperplane misses the object) are valid — must not trigger the
  uniform-render health check during animation (pass-through flag)

| Subtask | Estimate |
|---------|----------|
| Mesh slicer (quad classification, polygon emission, triangulation) | 6h |
| IFS shader slice mode | 4h |
| DSL/CLI `slice-w` + animation + health-check handling | 3h |
| Tests (slice of tesseract = cube at w₀=0), reference images, docs | 3h |

---

## F3: Distance-Estimator Fractals — ~30h

Ray-marched (sphere-traced) escape-time fractals: Mandelbulb, quaternion Julia sets
(genuinely 4D — slices/rotations reuse the project's 4D controls), Mandelbox.
Subsumes the TODO item "julia sets over ℂ".

**Implementation notes:**
- New IS program family: sphere tracing against a distance estimator
  `DE(p) = 0.5·|z|·log|z| / |z'|` (escape-time DE); bounding-sphere entry, max-step
  and epsilon-by-distance termination (cone tracing epsilon scaling)
- Normals via 4-tap tetrahedral gradient of the DE; existing material model applies
- Mandelbulb: power-n formula (default n=8), iteration count, bailout as parameters
- Quaternion Julia: `z ← z² + c` over quaternions; `c` is a Vec4 parameter; the
  rendered set is the intersection with a 3-space — wire `rot-x-w/y-w/z-w` to rotate
  the slicing 3-space (consistent with the rest of the 4D UX)
- Mandelbox: scale/min-radius/fold parameters; shares the framework
- Orbit-trap coloring: track min distance to trap shapes during iteration, map via
  a colormap (cheap, dramatic; coordinates with Sprint 36 colormaps if available,
  otherwise a fixed built-in ramp)
- Performance: target <400 ms/frame at 800×600 like the 4D IFS fractals; expose
  `max-steps`, `iterations` to trade quality

| Subtask | Estimate |
|---------|----------|
| DE sphere-tracing IS framework (entry, stepping, normals, epsilon) | 8h |
| Mandelbulb | 4h |
| Quaternion Julia + 4D rotation integration | 6h |
| Mandelbox | 3h |
| Orbit-trap coloring | 3h |
| DSL/CLI, tests, reference images, docs | 6h |

---

## F5: Procedural Sun-Sky (Hosek-Wilkie) — ~16h

Analytic clear-sky radiance model replacing the need for HDR files: parameterized by
sun elevation/azimuth, turbidity, and ground albedo. Feeds the existing env-map + IBL
pipeline; animatable (sunset time-lapse over a sponge).

**Implementation notes:**
- Port the Hosek-Wilkie reference implementation (check license of the reference
  code before porting; reimplementing from the paper's coefficient tables is the
  fallback — coefficients are published data)
- Evaluate into an equirectangular float buffer at scene load (e.g. 1024×512);
  register it through the exact same path as a loaded `.hdr` — IBL importance
  sampling, tone mapping, and background sampling all work unchanged
- Add the sun itself as an explicit directional light (the analytic model's solar
  disc is too small for stable IBL sampling): direction from azimuth/elevation,
  intensity/color from the model's solar radiance, scaled consistently
- DSL: `envSky = Some(SunSky(elevationDeg, azimuthDeg, turbidity = 3f,
  groundAlbedo = 0.3f, intensity = 1f))` — mutually exclusive with `envMap`
- Animation: changing sun parameters re-evaluates the buffer per frame (~ms-scale);
  reuse the Sprint 27 in-place env-texture update to avoid scene rebuilds

| Subtask | Estimate |
|---------|----------|
| Model port + coefficient validation against published reference images | 6h |
| Env-map pipeline integration (buffer, IBL, tone mapping) | 3h |
| Sun directional light + DSL/CLI + animation path | 3h |
| Tests (radiance sanity at known sun angles), reference images, docs | 4h |

---

## F6: Audio-Reactive Animation — Offline — ~24h

Analyze an audio file, derive per-frame parameter envelopes, render the animation with
the existing system, and mux the audio into the output video. The analysis and mapping
machinery is reused by F7.

**Implementation notes:**
- Audio decode through libav (already a dependency since Sprint 27): decode to mono
  float PCM; windowed FFT (2048 samples, hop = frame interval) — JVM-side, e.g.
  JTransforms or a small hand-rolled radix-2 (no native code needed)
- Features per frame: band energies (configurable bands, default bass/mid/treble),
  RMS loudness, spectral flux onset detection (beat events)
- Smoothing: attack/release envelope follower per binding (raw FFT is too jittery
  to drive parameters directly)
- Mapping DSL:
  ```scala
  audio = Some(AudioTrack("track.flac", bindings = List(
    Band(20, 150)   -> Param.Level(min = 2f, max = 4f, attack = 0.05, release = 0.3),
    Band(150, 2000) -> Param.RotXW(degreesPerSecondAt(1.0) = 90f),
    Onset           -> Param.PaletteStep,
  )))
  ```
  Bindings target the same parameter space as `--animate` (level, 4D rotations,
  film-thickness, fog density, light intensity, IBL strength)
- Mux: extend `VideoEncoder` to copy the audio stream into the MP4/MKV output
  (offset-aligned to frame 0)
- Determinism: same file + same bindings → identical frames (no RNG)

| Subtask | Estimate |
|---------|----------|
| Audio decode + FFT pipeline | 6h |
| Feature extraction (bands, RMS, onsets) + envelope smoothing | 5h |
| Mapping DSL + integration with animation parameter system | 6h |
| Audio muxing in VideoEncoder | 3h |
| Demo scene, tests (synthetic tones → known envelopes), docs | 4h |

---

## F7: Audio-Reactive Animation — Real-Time — ~26h

Live audio (microphone or system loopback) drives scene parameters in the interactive
preview. Both sources via one PipeWire capture backend (decided 2026-06-10); the
source is a runtime option.

**Implementation notes:**
- PipeWire capture via JNI (small C shim against `libpipewire-0.3`) or the
  `pw-cat --record -` subprocess as a zero-JNI fallback (spike both, prefer the
  subprocess if latency is acceptable — it removes a native dependency)
- Streaming analysis: ring buffer + the F6 FFT/feature code with shorter windows
  (1024) for latency; target audio→parameter latency < 50 ms
- Parameter application: bind only to **fast-update paths** — 4D rotation/projection
  (GAS refit, ~ms), light intensity, IBL strength, fog density. Level changes force
  scene rebuilds (hundreds of ms) — allow but document the jank; F17's accumulation
  reset is the natural integration point (each parameter change resets accumulation)
- Without F17, a degraded mode works (single-sample frames, no accumulation), but
  the intended experience needs F17's progressive loop
- Source selection: `--audio-input mic|loopback|default` + device name

| Subtask | Estimate |
|---------|----------|
| PipeWire capture backend (spike subprocess vs. JNI, then implement) | 8h |
| Streaming FFT/feature pipeline (ring buffer, low-latency windows) | 4h |
| Live binding to fast-update parameter paths | 6h |
| F17 integration (accumulation reset semantics, frame pacing) | 4h |
| Latency measurement harness, docs, demo | 4h |

---

## F8: Adaptive Variance-Based Sampling — ~18h

Concentrate accumulation samples where pixel variance is high instead of uniformly.
Speeds up IBL, area-light, DoF (Sprint 35), and dispersion (Sprint 32) renders at
equal quality.

**Implementation notes:**
- Track per-pixel running mean and M2 (Welford) across accumulation frames in two
  extra float buffers
- After a warm-up of N uniform frames (default 4), subsequent launches process only
  pixels whose relative variance exceeds a threshold; converged pixels are frozen
  (their value no longer changes — deterministic and artifact-free since frozen
  pixels keep their accumulated mean)
- Implementation in raygen: a per-pixel active mask buffer; early-exit costs one
  read per ray — measure overhead on fully-active scenes (<2 % acceptable)
- Termination: stop the whole render early when <0.5 % of pixels remain active —
  this turns `accumulation = 64` into a budget, not a fixed cost
- `--stats` reports active-pixel ratio per frame and effective speedup
- Interplay with the denoiser (Sprint 29): denoising prefers uniform noise—document
  that adaptive + denoise should use a lower variance threshold

| Subtask | Estimate |
|---------|----------|
| Variance buffers (Welford) + active-mask plumbing | 4h |
| Adaptive raygen loop + early-out + freeze semantics | 6h |
| CLI/DSL (`RenderSettings.adaptive`), stats reporting | 3h |
| Tests (deterministic freeze, equal-quality-vs-uniform image checks), docs | 5h |

---

## F10: Vector Field Visualization — ~26h

Visualize `f(x,y,z) → Vec3`: arrow glyphs, streamlines, and streamtubes, colored by
magnitude. Builds on Sprint 36's field evaluation and colormaps and Sprint 29's curves.

**Implementation notes:**
- Field definition mirrors Sprint 36's scalar fields (same expression mechanism,
  three components); built-in demo fields: dipole, vortex, Lorenz flow
- Glyphs: instanced arrow = cone + cylinder per seed point on a regular grid inside
  the bounds; length/color by magnitude (colormap); existing instancing path
- Streamlines: RK4 integration CPU-side from a seed grid (configurable density),
  step-size control by local magnitude, max length/steps; output polylines
- Streamtubes: streamline polylines → Sprint 29 curve primitives with radius
  optionally scaled by magnitude
- DSL: `VectorField(expression | preset, bounds, viz = Glyphs(spacing) |
  Streamlines(seeds, maxSteps) | Streamtubes(...), colormap)`
- Animation: time-dependent fields `f(x,y,z,t)` re-integrate per frame (CPU cost
  documented; start with static fields)

| Subtask | Estimate |
|---------|----------|
| Vector field evaluation + DSL types + presets | 5h |
| Glyph renderer (instanced arrows, magnitude coloring) | 5h |
| Streamline integrator (RK4, seeding, termination) | 6h |
| Streamtubes via curve primitive | 4h |
| Tests (integrator vs. analytic flow), reference images, docs | 6h |

---

## F11: Desktop Lens Window — ~20h

The interactive window becomes a "lens": capture the desktop region behind the window
and composite the rendered objects over it, so a tesseract appears to float over your
desktop. From TODO ("capture background by reading the desktop below the window").

**Implementation notes:**
- Capture backends:
  - X11: XComposite/XShm grab of the root window region under the window — direct
    and fast
  - Wayland: no direct grab; use the xdg-desktop-portal ScreenCast API (PipeWire
    video stream of the output), then crop to the window rect. Requires a one-time
    user permission dialog — unavoidable, document it
- The captured frame becomes the background plate: upload via the Sprint 27
  in-place `updateTexture` path into a dedicated background slot sampled by the miss
  shader (replaces background color/env map while active)
- Window-move/resize tracking: re-crop per frame; throttle capture to display
  refresh
- Compositing correctness: the plate is *behind* the scene only — reflections/
  refractions of the desktop in glass objects come free via the miss shader, which
  is the visual payoff
- Sequencing: X11 backend first (the user's system runs X11-capable Ubuntu);
  Wayland portal second

| Subtask | Estimate |
|---------|----------|
| X11 capture backend (XShm region grab, window tracking) | 6h |
| Wayland portal backend (ScreenCast + crop) | 6h |
| Background-plate texture path (miss-shader slot, per-frame update) | 4h |
| Interactivity polish (throttling, pause when occluded) | 2h |
| Tests (X11 under xvfb with a known root pattern), docs | 2h |

---

## F12a: Stereoscopic Still/Video Rendering — ~14h

Render stereo pairs for 3D displays and VR viewers: side-by-side, over-under,
anaglyph, and separate-file output. Pure camera-rig math — no new rendering tech.

**Implementation notes:**
- Stereo rig: eye separation (interocular distance, world units) along the camera
  right vector; convergence via asymmetric frustum shift at a configurable
  convergence distance (NOT toe-in rotation — toe-in causes vertical parallax)
- Frustum shift needs a small raygen extension (horizontal offset term), reused by
  F12b later
- Output modes: `sbs` (double-width), `ou` (double-height), `anaglyph` (red/cyan
  channel merge), `pair` (`_L`/`_R` files); video encoding works unchanged on the
  composite frames
- DSL: `camera.stereo = Some(Stereo(eyeSeparation = 0.065f, convergence = 3f,
  output = StereoOutput.SideBySide))`; CLI mirrors it
- Render cost: exactly 2× — document

| Subtask | Estimate |
|---------|----------|
| Stereo rig math + asymmetric frustum in raygen | 4h |
| Output modes (compositing, anaglyph merge, file pairs) | 4h |
| CLI/DSL wiring | 2h |
| Tests (L/R differ, parallax direction correct), reference images, docs | 4h |

---

## F12b: VR Live Preview (OpenXR) — ~40h

View the interactive scene in a headset. Honest assessment: the renderer currently
produces frames in hundreds of ms; a headset needs ~90 Hz per eye. This feature is
only viable on top of F17 (progressive preview) with aggressive resolution scaling,
and ships as "comfortable orbiting of a converging render", not full ray tracing at
headset rate.

**Implementation notes:**
- OpenXR session via JNI shim (instance, session, swapchains per eye, frame loop);
  Monado for desktop testing without hardware
- Architecture: render thread produces progressive frames into a shared texture;
  the OpenXR compositor thread re-projects the latest converged image per eye at
  headset rate (async reprojection of a quad/skybox layer — head rotation stays
  smooth even when the render lags; translation reveals staleness, acceptable for
  inspection use)
- Stereo from F12a's rig; per-eye view matrices from OpenXR pose data
- Camera motion resets accumulation (F17 semantics); a resolution scale (e.g.
  0.5× per eye) trades sharpness for convergence speed
- Risk: this is the highest-uncertainty item in the backlog; budget includes a 6h
  feasibility spike whose outcome may revise the whole estimate

| Subtask | Estimate |
|---------|----------|
| Feasibility spike (Monado, swapchain interop with CUDA/OptiX output) | 6h |
| OpenXR session/frame loop JNI shim | 12h |
| Progressive-render → compositor layer architecture | 10h |
| F12a/F17 integration (per-eye rig, accumulation reset, resolution scale) | 8h |
| Comfort/safety polish, docs | 4h |

---

## F15: Volumetric Lighting / God Rays — ~24h

Single-scattering participating media: light shafts streaming through the holes of a
Menger sponge. The most sponge-specific visual upgrade in the backlog.

**Implementation notes:**
- Extend the fog model (Sprint 21.7) from pure absorption to absorption +
  in-scattering: march the camera ray through the medium (jittered fixed steps,
  default 32); at each step cast a shadow ray to each shadow-capable light and
  accumulate `T(s) · σs · phase(θ) · L_light · exp(−σt·d_light)`
- Phase function: Henyey-Greenstein with asymmetry `g` (0 = isotropic, 0.6 ≈ hazy
  forward scatter)
- Cost control: shadow rays per step are the budget — march only up to the first
  surface hit, jitter steps per pixel (noise → handled by accumulation + denoiser),
  optional max-distance
- Directional lights give the classic crepuscular rays; F5 (sun-sky) is the ideal
  source but any directional/point light works
- Coordination: Sprint 36.3 (volume rendering) builds a density-field marcher; this
  feature is the *lit-medium* sibling. Whichever lands second reuses the marching
  loop of the first — if F15 is prioritized before Sprint 36, design the loop for
  both (density sampling callback)
- DSL: `fog = Some(Fog(density, color, scattering = Some(Scattering(g = 0.3f,
  steps = 32))))` — backward compatible: no `scattering` = current absorption-only

| Subtask | Estimate |
|---------|----------|
| Scattering march in the miss/hit integration path | 8h |
| Shadow-ray sampling per step + jitter + budget controls | 6h |
| DSL/CLI (Scattering params), backward compatibility | 3h |
| Reference images (sponge god rays, fog + area light), integration/manual tests | 4h |
| Docs + arc42 note (shared marching loop contract with Sprint 36) | 3h |

---

## F17: Real-Time Progressive Preview — ~26h

The interactive OptiX window accumulates samples continuously, resets on camera/scene
changes, and shows a denoised in-progress image. Transforms interactive exploration
and is the hard dependency of F7, F12b, and B-PARAMEXP.

**Implementation notes:**
- Accumulation already exists for headless renders; the work is making it
  *interactive*: render thread accumulates into the HDR buffer indefinitely;
  any invalidation (camera, 4D rotation, material, light, scene edit) resets the
  frame counter
- Invalidation triggers: centralize "scene/view changed" signaling — the camera
  handlers and 4D-rotation handlers currently trigger re-render directly; route
  them through one dirty-flag mechanism (small refactor of the input → engine path)
- Denoised display: every N accumulated frames (adaptive: 1, 2, 4, 8…), run the
  Sprint 29 denoiser into the display buffer while accumulation continues — the
  image visibly "settles" instead of staying noisy
- Threading: input handling must stay responsive while frames render; the current
  per-frame render call blocks — move the render loop off the input thread
  (LibGDX-side constraint check; the engine traits from Sprint 17 isolate this)
- Convergence HUD: frame count + active-pixel ratio (if F8 exists) overlay, toggle
  with a key
- This is also where DLSS-RR/NRD-style real-time denoising *would* slot in if ever
  revisited; the OptiX denoiser is sufficient for the inspection use case

| Subtask | Estimate |
|---------|----------|
| Interactive accumulation loop + reset semantics | 8h |
| Centralized invalidation (input → dirty-flag refactor) | 4h |
| Denoiser-in-the-loop (cadence, display buffer) | 4h |
| Render/input threading separation | 6h |
| HUD, tests (reset on each input class), docs | 4h |

---

## F18: Non-Euclidean / Quotient-Space Rendering — ~40h (2 sprints)

Render the *inside* of compact flat 3-manifolds: the 3-torus and its twisted variants
(the exact list already in TODO.md — half-turn, quarter-turn, third/sixth-turn prisms,
Hantzsche-Wendt). One sponge's worth of geometry appears as an infinite lattice of
images of itself. Hyperbolic space is explicitly out of scope (separate future item).

**Implementation notes:**
- Principle: the scene lives in a fundamental domain (a box or prism). A ray that
  exits a face re-enters through the identified face, transformed by the gluing
  isometry (translation, possibly composed with a rotation/flip). Iterate up to a
  wrap budget (e.g. 64 wraps) — effectively unrolling the universal cover
- Implementation: wrap loop in raygen around `optixTrace` — trace within the domain
  (domain walls as max-t), on miss apply the face isometry to origin/direction and
  re-trace. No shader changes for geometry; lights must be wrapped too (shadow rays
  follow the same rule — this is the subtle part: light paths through k wraps mean
  k-fold images of each light; cap light wraps separately, default 2–3)
- The 10 flat manifolds differ only in their face-identification isometries:
  a preset table (orientable six first; non-orientable need a parity flip in the
  isometry — handle normals accordingly)
- Orientation/visibility sanity: start with the 3-torus (pure translations — no
  twist), verify against the trivial expectation (periodic tiling), then add twists
- Fog (existing) is essential for depth perception in infinite vistas; document
  recommended settings
- 4D quotient spaces are deliberately out of scope

| Subtask | Estimate |
|---------|----------|
| Wrap loop in raygen (domain bounds, isometry application, budget) | 10h |
| 3-torus preset + correctness validation (periodicity tests) | 4h |
| Twisted/prism/Hantzsche-Wendt isometry table + non-orientable handling | 8h |
| Wrapped lighting/shadows (light image budget) | 6h |
| DSL/CLI presets (`manifold = Manifold.ThreeTorus`, domain size) | 4h |
| Reference images, integration/manual tests, user-guide section, arc42 | 8h |

---

## F19: Quasicrystals via Cut-and-Project — ~22h

Generate 3D icosahedral quasicrystals (Penrose-tiling analogs) by projecting a slice
of the 6D hypercubic lattice ℤ⁶. Mathematically the same "project from higher
dimensions" idea the whole project is built on, applied to aperiodic order.

**Implementation notes:**
- Cut-and-project: split ℝ⁶ into a 3D "parallel" (physical) and 3D "perpendicular"
  space via the icosahedral projection matrices; a lattice point is accepted iff its
  perpendicular projection lands inside the acceptance window (rhombic
  triacontahedron). Enumerate ℤ⁶ points in a radius-bounded box (cheap pruning via
  perpendicular-norm bound)
- Geometry: accepted points → vertices (spheres); edges connect pairs whose
  parallel-space distance equals the canonical edge length (within ε) → struts as
  cylinders or Sprint 29 curves. Renders as the classic icosahedral framework
- Animation: the window offset γ ∈ ℝ³ (perpendicular-space translation of the cut)
  morphs the structure — vertices pop in/out; animate via `--animate gamma=...`
  (discrete pops; document that smooth morphing is inherent to the math, not a bug)
- Size control: physical-space radius parameter; vertex/strut radii; expect ~10³–10⁴
  instances at radius 5 — fine for the existing instancing path
- 2D Penrose (pentagonal, from ℤ⁵) as a near-free bonus preset rendered as a tiling
  on a plane — optional stretch

| Subtask | Estimate |
|---------|----------|
| 6D lattice enumeration + icosahedral projection + acceptance test | 8h |
| Geometry emission (vertices, edge detection, struts) | 4h |
| Window-offset animation | 3h |
| DSL/CLI (`Quasicrystal(radius, gamma, vertexRadius, strutRadius)`) | 3h |
| Tests (vertex counts vs. published densities, inflation symmetry spot checks), docs | 4h |

---

## F20: SDF Combinators — CSG and Fractal Morphing — ~16h

Boolean and blending operations on distance-estimator shapes (F3): smooth
union/intersection/difference and parametric morphing — subtract a sphere from a
Mandelbulb, morph a Menger sponge into a quaternion Julia set.

**Implementation notes:**
- Requires F3's DE framework; primitive DEs additionally needed: sphere, box,
  torus (trivial analytic SDFs), plus the existing IFS sets reformulated as DEs
  where applicable (Menger sponge has a classic analytic DE — add it here)
- Combinators: `min` (union), `max` (intersection), `max(a,−b)` (difference),
  polynomial smooth-min `smin(a,b,k)`, and `lerp(a,b,t)` for morphing (t animatable)
- Evaluation: a small expression tree (max depth ~8, max ~16 nodes) encoded into
  launch params; the IS program interprets it per DE evaluation — avoids PTX
  recompilation per scene. Node = {op, child indices | leaf shape id + params}
- Lipschitz caveat: smooth-min and morph break the DE's distance bound — multiply
  step size by a safety factor (0.7) when those ops are present
- Material: single material per combined object in v1 (per-leaf materials need
  blending rules — defer)
- DSL: `SdfObject(Morph(MengerDE(level = 4), Mandelbulb(power = 8), t = "t"))`

| Subtask | Estimate |
|---------|----------|
| Expression-tree encoding + IS interpreter | 6h |
| Primitive SDF leaves (sphere/box/torus/Menger DE) | 3h |
| Smooth-min/morph + step-size safety | 2h |
| DSL tree builder, animation of `t`, tests, reference images, docs | 5h |

---

## F21: Gravitational Lensing / Black Hole — ~30h

Bend camera rays along null geodesics around a Schwarzschild mass: a black hole with
photon ring, Einstein-ring lensing of the env map, and an optional accretion disk —
or a Menger sponge gravitationally lensed. Connects to the "4D spacetime" TODO thread.

**Implementation notes:**
- Geodesic integration in the equatorial-symmetric form: integrate the photon
  trajectory in the plane spanned by ray origin/direction and the mass center
  (Schwarzschild geodesics are planar — reduces to a 2D ODE), e.g. RK4 on
  `u'' + u = 3GM·u²` (u = 1/r) with adaptive steps near the photon sphere (r = 3GM)
- Hybrid handoff: integrate only within an influence radius (e.g. 50 GM); when the
  ray exits, hand the (bent) straight ray to the normal OptiX trace / env-map miss.
  Rays crossing the horizon (r < 2GM) terminate black
- Scene objects inside the influence region: after each integration step, march-test
  against the OptiX scene with short straight segments (segment length = step) —
  correct and simple, cost proportional to steps (~100–300 per affected ray)
- Accretion disk: textured annulus in the equatorial plane (3GM–12GM), procedural
  turbulence; optional approximate Doppler beaming/redshift via a color-temperature
  shift by tangential velocity — visually convincing without full GR radiative
  transfer
- This is a special camera/integrator mode (`lensing = Some(BlackHole(mass, pos))`),
  not a material — document the model's limits (single mass, static, no Kerr spin)

| Subtask | Estimate |
|---------|----------|
| Planar geodesic integrator (RK4, adaptive near photon sphere) | 10h |
| Hybrid handoff (influence sphere, segment tracing against scene) | 6h |
| Accretion disk + approximate Doppler/redshift coloring | 6h |
| DSL/CLI, demo scenes (env-map lensing, lensed sponge) | 3h |
| Tests (deflection angle vs. analytic weak-field formula), reference images, docs | 5h |

---

## B-WYTHOFF: Wythoff Construction (Uniform Polytopes) — ~45h (2 sprints)

Full kaleidoscopic construction of uniform polyhedra and polychora from ringed Coxeter
diagrams: all Archimedean solids, prisms/antiprisms, and the 4D uniform families.
Supersedes T-SEMI when built.

**Implementation notes:**
- Generate the symmetry group by reflections in the fundamental simplex mirrors
  (Coxeter group; finite groups A/B/H in ranks 3–4), orbit a seed point chosen by
  the ringed nodes (Wythoff vector), deduplicate vertices, reconstruct faces/cells
  by orbiting sub-diagrams
- Builds directly on Sprint 38's Schläfli generator (38.2) — the reflection-group
  machinery generalizes it; plan as an extension, not a parallel implementation
- Face reconstruction (which orbits form which faces) is the genuinely hard part;
  use the standard coset-of-parabolic-subgroups method
- Snubs (alternated, chiral) are a stretch goal — they need even-subgroup handling
- DSL: `UniformPolytope(coxeter = "x3o5x")`-style linear notation

| Subtask | Estimate |
|---------|----------|
| Coxeter group generation + orbit machinery (rank 3–4) | 12h |
| 3D uniforms (Archimedeans, prisms/antiprisms) + face reconstruction | 10h |
| 4D uniforms + projection integration | 12h |
| DSL/CLI notation parser | 4h |
| Tests (vertex/face counts vs. published data for all 13 Archimedeans), docs | 7h |

---

## B-ROTOPES: Rotopes — ~30h + design spike

Shapes generated by composing extrusion, tapering, and rotation operations across
dimensions (duocylinder, spherinder, etc.).

**Implementation notes:**
- Needs a 6h design spike first: representation question — closed-form parametric
  surfaces (then this largely reduces to Sprint 37's framework plus an operation
  algebra) vs. mesh-level operators (extrude/taper/rotate acting on meshes)
- Recommended direction after spike: operation algebra producing parametric patches
  in 4D, rendered through Sprint 37's pipeline; duocylinder and spherinder as the
  validation cases
- Estimate is indicative (~30h) until the spike lands

---

## B-SSS: Subsurface Scattering — ~28h

Translucent materials (wax, marble, skin-like) where light enters, scatters inside,
and exits elsewhere.

**Implementation notes:**
- In a Whitted-style pipeline the practical choice is the normalized-diffusion BSSRDF
  approximation (Christensen-Burley): at a hit on an SSS material, sample N exit
  points around the entry by the diffusion profile (radius parameter per channel),
  probe with short rays, gather lighting at exit points
- Parameters: `mfp` (mean free path per RGB), `scale`; preset `wax`, `marble`,
  `jade`
- Noise behaves like area lights — accumulation + denoiser (Sprint 29) absorb it
- True volumetric random-walk SSS is out of scope until/unless a path-traced GI mode
  exists (deprioritized)

| Subtask | Estimate |
|---------|----------|
| Diffusion-profile sampling + exit-point probing in closest-hit | 10h |
| Material parameters/presets + DSL/CLI | 4h |
| Lighting gather at exit points (reuse direct-lighting code) | 6h |
| Tests, reference images (backlit wax sphere), docs | 8h |

---

## B-DATASETS: VTK/NetCDF Dataset Import — ~30h

Render real scientific data with the Sprint 36 (volume/isosurface) and F10 (vector)
machinery instead of analytic expressions.

**Implementation notes:**
- Readers: NetCDF via the `netcdf-java` library (mature, pure JVM); VTK legacy
  structured-points format hand-parsed (the subset is simple ASCII/binary headers) —
  full VTK XML is out of scope
- Regular/rectilinear grids only in v1; scalar and 3-component vector variables
- 3D texture upload path (Sprint 36 task 36.2 builds it for analytic fields
  evaluated to grids — datasets reuse it directly); large volumes: max-dimension
  downsampling with a warning (e.g. cap 512³)
- Time-series variables map to animation `t` (nearest or lerp between time steps)
- DSL: `Dataset("file.nc", variable = "temperature")` usable wherever
  `ScalarField`/`VectorField` expressions are

| Subtask | Estimate |
|---------|----------|
| NetCDF reader + variable/grid mapping | 8h |
| VTK structured-points parser | 5h |
| Grid → 3D texture bridge + downsampling | 6h |
| Time-series → animation binding | 4h |
| Tests (synthetic .nc/.vtk fixtures), demo with a public dataset, docs | 7h |

---

## B-PARAMEXP: Multi-Dimensional Parameter Exploration — ~20h

Interactively vary 2–3+ parameters (fractal level, 4D rotations, material values)
with immediate visual feedback. Needs F17 for the feedback loop to feel live.

**Implementation notes:**
- Keyboard-first UI (consistent with the existing interactive controls): select a
  parameter (number keys), adjust (+/-/arrows with step modifiers), on-screen
  parameter HUD; no GUI toolkit dependency
- Parameter registry: expose the same parameter space as `--animate` bindings
  (shared with F6/F7's mapping targets — build the registry once, three features
  use it)
- Fast paths (4D rotation, lights, materials) update live; rebuild-requiring
  parameters (level) show a "pending" indicator and apply on release
- Snapshot key dumps the current parameter set as a ready-to-run CLI line/DSL
  snippet (subsumes the old "view bookmarking" idea)

| Subtask | Estimate |
|---------|----------|
| Parameter registry (shared with audio bindings) | 6h |
| Keyboard interaction + HUD overlay | 6h |
| Fast-path vs. rebuild-path application + pending indicator | 4h |
| Snapshot-to-CLI/DSL dump, tests, docs | 4h |

---

## T-MTLX: MaterialX Support, Layers 1–3 — ~70h (3 sprints)

Industry-standard material distribution format. The detailed layer plan lives in
TODO.md; summarized here with estimates. Sprint 33 (PBR texture sets) builds the
map slots Layer 3 needs.

| Layer | Scope | Estimate |
|-------|-------|----------|
| 1 | MaterialX C++ SDK as CMake dep; `MtlxLoader.cpp` extracting Standard Surface / OpenPBR parameters; JNI + `MtlxMaterial` case class | ~24h |
| 2 | Parameter mapping onto `InstanceMaterial` (base_color, roughness, metallic, IOR, emission); `--objects ...:mtlx=path` wiring | ~20h |
| 3 | `<image>` node resolution against `--mtlx-texture-dir`; upload via texture pipeline into the Sprint 33 map slots | ~26h |

Out of scope (unchanged from TODO.md): coat, sheen, subsurface, anisotropy, node-graph
evaluation, color management. Unsupported inputs warn and are ignored.

---

## T-SOR: Surface of Rotation — ~10h

Revolve a user-defined profile curve around an axis. Cheap once Sprint 37's parametric
framework exists (a SOR is `f(u,v) = (r(v)·cos u, h(v), r(v)·sin u)`).

| Subtask | Estimate |
|---------|----------|
| Profile curve DSL (piecewise linear + optional Catmull-Rom through points) | 3h |
| SOR as parametric specialization (Sprint 37 mechanism) | 3h |
| Presets (vase, torus-by-revolution validation), tests, docs | 4h |

---

## T-STAR: Regular Star 4-Polytopes — ~14h

The Schläfli-Hess star polychora ({5/2,5,3} etc.) via the Sprint 38 generator extended
to fractional Schläfli symbols.

**Implementation notes:**
- Vertices of all ten star polychora coincide with 120-cell/600-cell vertex sets —
  generation is about *face structure*, not new vertex math
- Self-intersecting faces are fine for a ray tracer (no winding/CSG needed —
  render the geometric surface as-is); document that interiors are not
  density-correct
- Start with {5/2,5,3} (small stellated 120-cell) and {3,3,5/2}; full set of ten as
  data-driven extensions

| Subtask | Estimate |
|---------|----------|
| Fractional-symbol handling in the generator (38.2 extension) | 6h |
| First two star polychora + projection verification | 4h |
| Remaining symbols (data), tests (cell counts), reference images, docs | 4h |

---

## T-SEMI: Semiregular Polyhedra/Polytopes — ~16h

Interim hardcoded approach until B-WYTHOFF exists (which supersedes this).

| Subtask | Estimate |
|---------|----------|
| 13 Archimedean solids from published vertex data (data + mesh assembly) | 8h |
| 4D prisms/antiprisms (extrusion of 3D polyhedra along w) | 4h |
| Tests (V/E/F counts), DSL/CLI names, docs | 4h |

---

## T-CFUNC: Functions ℂ→ℂ — ~12h

Visualize complex functions: domain coloring (argument → hue, modulus → brightness)
as a procedural plane texture, and Re/Im/|f| height surfaces via the parametric
framework (Sprint 37).

| Subtask | Estimate |
|---------|----------|
| Complex expression evaluation (reuse Sprint 36 expression mechanism, complex ops) | 4h |
| Domain-coloring procedural texture mode | 3h |
| Height-surface mode via parametric specialization | 2h |
| Presets (z², 1/z, Möbius, sin z), tests, docs | 3h |

---

## T-HOPF: Hopf Fibration / 3-Sphere — ~16h

Render the Hopf fibration: fibers of S³ → S² as linked circles in 4D, projected
through the existing 4D pipeline as curve tubes (Sprint 29). The canonical "visualize
the 3-sphere" demo.

**Implementation notes:**
- For each base point (θ,φ) on S², the fiber is the circle
  `(cos(ξ)·e^{i(η+ψ)}, sin(ξ)·e^{iψ})` parameterized by ψ — emit as 4D control
  points, rotate/project with the standard 4D parameters, render as curves
- Base-point selection presets: latitude rings (the classic nested-tori picture),
  great-circle sweeps, random sets; color fibers by base-point position (hue by φ)
- Animation: sweep base points or rotate in 4D — both already supported parameter
  spaces

| Subtask | Estimate |
|---------|----------|
| Fiber generation + 4D projection of curve control points | 4h |
| Base-point presets + per-fiber coloring | 5h |
| DSL/CLI (`HopfFibration(rings, fibersPerRing, tubeRadius)`) | 3h |
| Tests (fibers lie on S³, linking sanity), reference images, docs | 4h |

---

## T-SPLAT: Gaussian Splats — ~50h + design spike

Render 3D-Gaussian-splatting scenes (.ply from 3DGS reconstructions). High effort and
an awkward fit: 3DGS is built for sorted rasterized alpha-blending, not ray tracing.

**Implementation notes:**
- 8h design spike first: (a) ray-traced ellipsoids with transmittance accumulation
  in an anyhit-style traversal (correct but potentially slow for 10⁶ splats),
  (b) preprocessed opacity/density volume (lossy), or (c) hybrid background-plate
  rendering (splats as env-like backdrop, ray-traced objects composited with depth)
- Use cases to validate: a reconstructed room as *environment* for a floating
  sponge (option c may win for this); spike decides
- Estimate indicative until the spike lands

---

## T-TRACE: 4D Spacetime Trace — ~30h + design spike

Extrude an animated 3D object through time as a 4D mesh (w = t), then view that
static 4D "worldtube" with the existing 4D rotation/projection — literally looking at
spacetime from outside.

**Implementation notes:**
- 4h design spike on representation: sample the animation at N time steps, connect
  consecutive 3D meshes (same topology required) into 4D prismatic cells, emit
  the boundary as the quad-based 4D mesh format the pipeline already projects
- Constraint v1: topology-stable animations only (rigid motion, rotation, scaling —
  not level changes)
- Demo: a tumbling cube's helical worldtube; a 3-sphere's worldtube = 4D ball slice

| Subtask | Estimate |
|---------|----------|
| Design spike (sampling density, cell construction, mesh size budget) | 4h |
| Animation sampling + 4D prism-cell boundary construction | 12h |
| Pipeline integration (existing 4D mesh path) + DSL | 6h |
| Demos, tests, docs | 8h |

---

## T-360MOVIE: Level-Sweep Movie with 360° Background — ~8h

Showcase, not infrastructure: fractional-level sweep (existing) + 360° video
background (Sprint 27) + optional soundtrack (F6 when available). Work is scene
design, render-farm time, and a docs/gallery entry.

| Subtask | Estimate |
|---------|----------|
| DSL scene + parameter tuning (level curve, camera path, background) | 4h |
| Render + encode + publish to website gallery | 3h |
| User-guide "making of" notes | 1h |

---

## T-SCALE / T-SHEAR: Per-Axis Scaling and Shearing — ~6h each

Affine-transform completions in the object spec.

**Implementation notes (both):**
- `scale=sx,sy,sz` and `shear=xy,xz,yx,yz,zx,zy` (or a 3×3 matrix shorthand) in
  `--objects` and DSL; compose into the existing per-instance transform
- The pitfall is normals: non-uniform scale/shear requires inverse-transpose normal
  transformation — verify the instance path handles it (OptiX object-to-world for
  normals); add a sheared-sphere lighting test that fails if normals are wrong
- Analytic IS primitives (sphere, cone): confirm they respect instance transforms
  (they do for rotation today; non-uniform scale turns spheres into ellipsoids —
  that is the feature, but test refraction through one)

| Subtask | Estimate (each) |
|---------|----------|
| Spec parsing + transform composition | 2h |
| Normal-transform verification + IS-primitive checks | 2h |
| Tests, reference image, docs | 2h |

---

## Housekeeping (from TODO.md, estimates only)

Sprint-filler items; no feature design needed:

| Item | Estimate |
|------|----------|
| Investigate `hs_err*.log` files | 2h |
| Investigate pending/ignored tests | 3h |
| Scala version used verbatim anywhere? | 1h |
| Polyhedra contract (analog of `Polytope4DContract`) | 4h |
| Scene/animation design guidance for the user guide | 4h |

(sbt-updates replacement is now Sprint 28 task 28.2.)
