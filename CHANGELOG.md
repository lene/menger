# Changelog

## [0.7.6] - 2026-06-26

### Added
- **OptiX final-frame denoising** — DSL `RenderSettings(denoise = DenoiseMode.Final)`
  and CLI `--denoise` denoise the final accumulated linear HDR frame before tone mapping.
- **OptiX curves primitive** — DSL `Curve(points, radius, material)` and CLI
  `type=curve:control-points=...:radius=...` render smooth swept tubes via the
  built-in round cubic B-spline primitive. Includes `TrefoilKnot` demo scene.
  Available in `optix-jni >= 0.1.5`.

### Changed
- **`optix-jni` 0.1.5** — switched from temporary source pin to published
  Maven Central artifact (includes denoiser API, curves primitive, and
  `updateTexture` from Sprint 27).

## [0.7.5] - 2026-06-15

### Added
- **Cross-repo quality standards** (`standards/`) — canonical scalafix, scalafmt,
  and hook scripts shared across menger, menger-common, and optix-jni; daily drift
  detection fails loudly when configs diverge.
- **Tiered hooks** — fast pre-commit (< 5 s for non-code changes) with branch guard,
  staged hygiene, and local standards parity; pre-push Phase 0 policy checks
  (`Test-Change:` trailer enforcement and rendering-discipline check) run before
  any compilation.
- **Local CI runner hardening** — gitlab-runner and GitHub Actions runner run as
  systemd services with auto-restart and resource caps; GPU jobs serialised with
  `limit = 1`; heartbeat alert fires if no runner picks up work in 24 h.
- **Multi-model AI review** — every MR receives structured reviews from Claude and
  DeepSeek posted as GitLab comments; disagreements preserved.
- **Release-on-merge automation** — merging an MR without `NORELEASE` label
  triggers `CreateRelease` → tag pipeline → GitHub mirror → install-smoke proof.
- **Enforcement audit** (`docs/ENFORCEMENT.md`) — every policy in AGENTS.md mapped
  to its enforcing mechanism; 8 gaps tracked as GitLab issues #155–#162.
- **GitLab merge checks** — pipelines must succeed and all discussions must be
  resolved before the merge button is enabled.
- **`docs/TESTING.md`** — test-failure protocol, flaky-test policy, and
  `Test-Change:` trailer requirement.
- **`docs/RENDERING.md`** — rendering-change discipline, sequential-mode rule,
  alpha-channel invariant, and new-feature checklist.
- **`--stats-json <path>`** CLI option — writes per-frame render statistics (frame
  time, ray counts, ms/Mray) as JSON on exit; enabled automatically with `--headless`.
- **`scripts/benchmark.sh`** — runs 4 representative scenes 3× each, computes
  median frame time, and compares against `scripts/perf-baseline.json` (15% threshold).
- **`PerfCheck` CI job** — advisory performance regression guard; runs on every push
  against the built deployable; `allow_failure: true` so it warns without blocking.
- **Renovate dependency automation** — `renovate.json` at repo root; groups sbt
  plugins and libraries separately; excludes pinned CUDA/OptiX/libav deps.
  Triggered at sprint close via `glab pipeline run`.
- **`Test:InstallSmoke` CI job** — unpacks the release zip and verifies `--version`
  and a headless render succeed on the packaged binary.
- **arc42 §10 G1–G7** — quality scenarios for the guardrail system.
- **arc42 §11 TR-13** — runner-availability risk and mitigations.

## [0.7.4] - 2026-06-14

### Added
- **`/arch-review`** — architectural review command (four axes: soundness, maturity,
  evolvability, performance architecture) with fitness-function nomination; first review
  in `ARCHITECTURE_REVIEW.md` and actionable backlog in `docs/ARCHITECTURE_BACKLOG.md`.
- **Video texture documentation and examples** (Sprint 27.10) — user guide and DSL
  reference coverage for rectangular video textures, 360-degree `EnvMapVideo`
  backgrounds, playback timing/repeat controls, IBL interaction, and performance
  recommendations. Added `examples.dsl.EnvMapVideoSponge` plus integration/manual
  coverage for env-map video playback.

### Changed
- **Minimum NVIDIA driver raised to ≥580.65 (CUDA 13).** The published `optix-jni`
  artifact (≥0.1.3) and `menger-geometry` native libs link the CUDA 13 runtime
  (`libcudart.so.13`). Older drivers fail at startup with CUDA error 35
  ("driver version is insufficient for CUDA runtime version"). See arc42 §2 (TC-4, TC-9)
  and `docs/TROUBLESHOOTING.md`. GPU CI runner hosts must also have `nvidia-persistenced`
  active (not masked) for the NVIDIA container toolkit to mount its socket.
- Bumped `optix-jni` dependency to **0.1.4** (CUDA 13.x toolkit pin — reproducible
  runtime ABI, deliberate minimum-driver floor).

### Fixed
- **4D dispatch in the non-interactive render path** — `sierpinski4d` and
  `hexadecachoron4d` were dispatched only in `InteractiveEngine`; `GeometryRegistry` /
  `RenderModeSelector` classified them as `Unsupported`, so Animation/Preview/Video
  engines (incl. `--animate`) failed with a misleading error. Added the missing dispatch
  branches + a completeness test; closed the `integration-tests.sh` / `manual-test.sh`
  parity gap for both types.
- Removed stale solved High Priority findings from `CODE_IMPROVEMENTS.md`.
- GitLab GPU jobs now install FFmpeg/libav packages without recommended dependencies,
  avoiding distro `libcuda1` shadowing the NVIDIA driver-mounted CUDA library.
- GitLab OptiX runtime jobs now explicitly request visible NVIDIA devices and
  fail fast when the GPU driver mount is missing instead of cascading render crashes.
- Scene builders now validate config-based, mixed-scene, and optimized 4D tracked
  build paths before any renderer instance allocation, closing `M-sceneb-validate-bypass`.
- Native scene instance IDs are now wrapped in an opaque app-level `InstanceId`;
  `-1` allocation failures fail scene builds at the boundary instead of being
  handled inconsistently by individual builders.

## [0.7.3] - 2026-06-08

### Fixed
- **GitLab OptiX runtime jobs** — `Run:UseDocker` and `CheckRunTime` now request
  graphics-capable NVIDIA driver mounts and refresh the RTX core linker setup before
  launching OptiX, matching the full test and integration jobs.

## [0.7.2] - 2026-05-31

### Changed
- **Package rename** (Sprint 24.2) — `menger.optix` → `io.github.lene.optix` in all Scala
  sources and JNI C++ symbols. Existing consumers of `optix-jni` must update imports.
- **`Params` → `BaseParams`** (Sprint 24.3) — OptiX launch parameter struct renamed in
  `OptiXData.h` and all C++ consumers. `MengerParams` (Sprint 24.5) extends `BaseParams`
  with 4D geometry and caustics fields; struct extension is layout-safe (offset 0 guarantee).
- **`optix-jni` decoupled from Menger geometry** (Sprint 25) — all 4D hit shaders
  (`hit_menger4d.cu`, `hit_sierpinski4d.cu`, `hit_hexadecachoron4d.cu`), `Project4D`,
  `CausticsRenderer`, and their JNI bindings moved to `menger-geometry`. `optix-jni` now
  contains zero Menger-specific types and is usable as a standalone GPU ray tracing library.

### Added
- **`menger-geometry` module** (Sprint 24.4–25.3) — in-repo sbt subproject for
  Menger-specific geometry. Hosts all 4D geometry, caustics, and `MengerRenderer`. Not
  published; depends on `optix-jni`.
- **`MengerRenderer`** (Sprint 25.3) — Scala class extending `OptiXRenderer`; routes all
  4D geometry `@native` calls through `libmengergeometry.so`. `menger-app` now uses
  `MengerRenderer` for all 4D scenes instead of calling `OptiXRenderer` directly.
- **`NativeOptiXApi`** (Sprint 25.4) — thin JNI bindings for the core OptiX pipeline:
  context, module, program groups (raygen/miss/hitgroup/triangle), and pipeline lifecycle.
  Handles are raw `Long` values; returns `0L` on failure. Lays groundwork for pipeline
  composition without the high-level `OptiXWrapper`.
- **Publication pipeline** (Sprint 24.6–24.7) — `menger-common` and `optix-jni` are now
  publishable to GitLab Package Registry and Maven Central (Sonatype Central Portal).
  CI jobs `PublishCommon` and `PublishOptixJni` trigger on tag. GPG signing via sbt-pgp.

### Fixed
- JNI `GetByteArrayElements`, `GetLongArrayElements`, `GetStringUTFChars` return values
  are now null-checked before use in `OptiXApiBindings.cpp`; prevents undefined behaviour
  under JVM memory pressure.
- JNI string and array resources released on every exit path (including C++ exception
  paths) in `OptiXApiBindings.cpp` and `MengerJNIBindings.cpp`.
- `createPipeline` rejects `groupHandles` arrays containing `0L` entries before calling
  the OptiX SDK, preventing a driver-level crash on failed program group creation.
- `FindClass` result null-checked before `ThrowNew` in all `MengerJNIBindings.cpp` catch
  blocks (regression of Sprint 22 fix re-introduced in new file).

---

## [0.7.1] - 2026-05-28

### Added
- **Image-Based Lighting (IBL)** (Sprint 23.1–23.4) — HDR env map illuminates objects via
  importance sampling and MIS balance heuristic. Enable with `Scene(ibl = Some(IBL(...)))`;
  requires `envMap` to also be set. `IBL(strength, samples)` controls intensity (default 1.0)
  and samples per shading point (1–8, default 1).
- **Multi-frame accumulation** (Sprint 23.5) — `RenderSettings(accumulation = N)` averages N
  independent renders with different random seeds, reducing IBL noise. Only applies in
  headless/render mode; interactive preview uses a single frame (zero performance regression).

---

## [0.7.0] - 2026-05-24

### Added
- **HDR environment maps in DSL** (Sprint 22.1) — `Scene(..., envMap = Some("panorama.hdr"))`
  renders an equirectangular HDR file as the scene background. Path resolved relative to
  `--texture-dir`. Background only; IBL is Sprint 23.
- **Tone mapping** (Sprint 22.2) — `Scene.toneMapping` controls HDR-to-display mapping:
  `ToneMapping.None` (clamp, default), `ToneMapping.Reinhard(exposure)` (smooth roll-off),
  `ToneMapping.ACES(exposure)` (Narkowicz 2015 filmic S-curve). Applied in miss shader to
  env map samples; exposure is a pre-tone-map multiplier (default 1.0).
- **`Sierpinski4D` DSL type** (Sprint 22.3) — `Sierpinski4D(level, material, projection, ...)`:
  4D pentachoron IFS fractal usable in DSL scenes, supports `edgeRadius`/`edgeMaterial`.
- **`FractalWithHDR` example scene** — animated glass `TesseractSponge` with level 1→4,
  XW/YW/ZW 4D rotation schedule, cliffside HDR background, Reinhard tone mapping.
- **`SierpinskiHDRRotation` example scene** — animated film `Sierpinski4D` with copper edges,
  simultaneous 3D Y-axis rotation + 4D XW rotation, cliffside HDR, Reinhard tone mapping.
- **`SceneConverter` moved to `menger.engines`** (Task 22.6) — resolves P0.A architecture
  layer violation; `menger.dsl` no longer imports `menger.config` or `menger.optix`.

### Fixed
- JNI `ThrowNew` no longer silently no-ops when `FindClass` returns null; all 11 throw sites
  now use a safe helper that guards against null class.
- JNI `setLights` loop now deletes 4 local refs per iteration, preventing local ref table
  exhaustion (JVM abort) with ≥4 lights.
- `setTriangleMeshNative` catch block now releases pinned JVM arrays on C++ exception, preventing
  pinned memory leak on mesh upload failure.
- `SceneObject.validateSceneMaterials` replaced by exhaustive `materialsToValidate` method on
  the sealed trait; new geometry types are now compile-time checked rather than silently skipped.

---

## [0.6.2] - 2026-05-22

### Added
- **4D Menger sponge analog** (`menger4d`) — iterative IFS ray traversal via custom OptiX
  intersection shader; O(1) VRAM regardless of level (level 10+ supported). Parameters:
  `level`, `dist-threshold`, `rot-xw`/`rot-yw`/`rot-zw`, `eye-w`, `screen-w`.
- **4D Sierpinski tetrahedron analog** (`sierpinski4d`) — IFS intersection shader; 4D
  recursive tetrahedral self-similar structure projected to 3D.
- **4D hexadecachoron sponge** (`hexadecachoron4d`) — IFS intersection shader based on the
  16-cell; distinct cross-section topology from `menger4d`.
- **Fractional levels for `sponge-recursive-ias`** — fractional `level` values (e.g. `2.5`)
  produce a cross-fade between adjacent integer IAS trees; smooth visual transition matching
  the menger4d/sierpinski4d fractional-level behaviour.
- **CLI animation via `--animate`** — `--objects` scenes now support `--animate
  "frames=N:param=start-end"` for headless multi-frame rendering; parameters include
  `rot-x-w`, `rot-y-w`, `rot-z-w`, `rot-x/y/z`, `level`, `projection-eye-w`,
  `projection-screen-w`. All 4D IFS types (`menger4d`, `sierpinski4d`, `hexadecachoron4d`,
  `sponge-recursive-ias`) are valid animation targets.
- **`RecursiveIAS` DSL sponge type** — `Sponge(spongeType = RecursiveIAS, level = ...)` in
  DSL scenes.
- **`SpongeLevelAnimation` example scene** — DSL animated scene sweeping `sponge-recursive-ias`
  level 1→4; demonstrates fractional-level cross-fade via `scene(t)`.
- **Fog / depth cue** (Task 21.7) — exponential distance-based attenuation for all geometry
  types. CLI: `--fog density=0.05:color=0.8,0.8,0.9`. DSL: `fog = Some(Fog(density = 0.05f))`.
  Zero overhead when disabled (`fog_density == 0`).
- **Image textures and PBR maps for cone and plane** (Task 21.6) — cone and plane geometry
  now support `texture`, `normal-map`, and `roughness-map` parameters. Cone uses cylindrical
  UV mapping; plane uses axis-aligned planar UV repeating per world unit. Wired via new
  `image_texture_index` field in `InstanceMaterial` (separate from `texture_index`, which
  indexes geometry data arrays for these types).

### Fixed
- Default plane no longer injected when `--plane` is omitted from CLI (was leaking a `y:-2`
  floor plane into every render). Integration tests updated with explicit `--plane y:-2` where
  the floor is intentional.
- `MeshFactory` and `InteractiveEngine` now accept canonical tesseract-sponge type names
  (`tesseract-sponge-volume`, `tesseract-sponge-surface`) in addition to legacy aliases.

---

## [0.6.1] - 2026-05-17

### Added
- **Image textures** (Sprint 20.1/20.2) — PNG/JPG/HDR/EXR texture loading for spheres and mesh
  geometry; spherical UV mapping for sphere hit shader; `--texture-dir` flag for base path
- **PBR normal and roughness maps** (Sprint 20.7) — `normal-map` and `roughness-map` per-object
  parameters; applied via `setMapTextures` in OptiX hit shaders
- **Procedural textures** (Sprint 20.4/20.8/20.12/20.13) — 11 GPU-computed types: ValueNoise,
  FBM, Worley, Gradient, Wood, Marble, LayeredNoise, XYZToRGB, HeatMap, Triplanar; all available
  for sphere, cone, plane, and mesh geometry via `procedural-type` / `procedural-scale` params
- **Environment map / IBL** (Sprint 20.3) — equirectangular HDR/EXR environment maps via
  `--env-map`; contributes to scene lighting (IBL) and background skybox
- **DSL texture syntax** (Sprint 20.10) — `texture`, `normalMap`, `roughnessMap`,
  `proceduralType`, `proceduralScale` exposed on all `SceneObject` DSL types

### Changed
- **ObjectSpec value objects** — grouped flat fields into typed value objects: `ObjectRotation`,
  `ConeGeometry`, `PlaneGeometry`, `ProceduralSpec`, `TextureMaps`; backward-compatible via
  forwarding accessors
- **OptiXRenderer facade split** — 995-line god class refactored into 5 responsibility traits
  (`OptiXSphereApi`, `OptiXMeshApi`, `OptiXPlaneApi`, `OptiXTextureApi`, `OptiXRenderApi`);
  external API unchanged
- **PlaneSpec / CheckerPattern value objects** — `addPlaneCheckerColorsWithMaterial` now accepts
  `PlaneSpec(axis, positive, value)` and `CheckerPattern(color1, color2)` overloads
- **Scene4DCache** — merged `anim4DState` / `cpu4DState` `AtomicReference` pair into single
  `Scene4DCache` in `InteractiveEngine`
- **CLI validator consolidation** — extracted `isDegree` validator and `requiresCausticsFlag`
  helper in `MengerCLIOptions`

### Removed
- Legacy `miss_plane.cu` plane intersection path (Sprint 20.5)
- Legacy CPU 4D projection path (`Mesh4D`, `RotatedProjection`) (Sprint 20.6)

---

## [0.6.0] - 2026-05-14

### Added
- **Platonic solids** (Sprint 19.1) — tetrahedron, octahedron, dodecahedron, icosahedron
  available as 3D primitives via `--objects type=tetrahedron|octahedron|dodecahedron|icosahedron`
- **Regular 4-polychora** (Sprint 19.2) — pentachoron (5-cell), hexadecachoron (16-cell),
  icositetrachoron (24-cell), hecatonicosachoron (120-cell), hexacosichoron (600-cell)
  via `--objects type=pentachoron|16-cell|24-cell|120-cell|600-cell`; share the same 4D
  projection parameters (`eye-w`, `screen-w`, `rot-x-w`, `rot-y-w`, `rot-z-w`) as tesseract
- **Cone** (Sprint 19.3) — analytical cone primitive (IS program, no triangle mesh) via
  `--objects type=cone`
- **Planes as first-class geometry** (Sprint 19.4) — planes rendered as scene objects with
  full material support (`--plane`, `--plane-material`)
- **Coordinate cross** (Sprint 19.5) — XYZ axis visualization as analytical cylinders;
  `--cross`, `--cross-length`, `--cross-thickness`, `--cross-material`; toggle with 'C' key
- **Geometry registry** (Sprint 19.6) — `ObjectType` central registry; adding a new geometry
  type requires registration only, not engine modification. See AD-21.
- **Per-object 3D rotation** (Sprint 19.7) — `rot-x`, `rot-y`, `rot-z` params in `--objects`
  spec (degrees); also exposed as `rotation: Vec3` on all DSL `SceneObject` types
- **Render time statistics** (Sprint 19.8) — `--stats` now reports ms/frame and ms/Mray
  in addition to ray counts

### Research Spikes
- **Max trace depth** (Sprint 19.9) — depth 8 is the practical ceiling for glass stacks at
  levels ≤ 5; no visual benefit beyond it. No follow-up scheduled.
  Findings: `docs/dev/sprint-19-spike-max-depth.md`
- **Fractional IAS sponge levels** (Sprint 19.10) — Approach B (two IAS trees: level N opaque
  + level N−1 at transparency f) confirmed viable; no shader or compositor changes needed.
  Scheduled for Sprint 21.4. Findings: `docs/dev/sprint-19-spike-fractional-ias.md`

---

## [0.5.8] - 2026-05-02

### Added
- **Multi-GAS Instance Acceleration Structure** (Sprint 18.1) — the top-level
  traversable is now an IAS; each scene object owns a private GAS referenced
  by an `OptixInstance`. Per-object materials, per-object IS programs, and
  nested instancing all become possible. See AD-17.
- **Recursive IAS Menger sponge** (Sprint 18.4) —
  `--objects type=sponge-recursive-ias:level=N` builds the sponge as nested
  IAS layers (one Level-1 GAS of 20 sub-cubes referenced 20 times per
  recursion). VRAM scales as O(N · 20) matrices instead of O(20ᴺ) triangles,
  enabling levels 6..14 (capped by `OPTIX_MAX_TRAVERSABLE_GRAPH_DEPTH`).
  See AD-20.
- **GPU 4D projection** (Sprint 18.3) — opt-in `--gpu-project-4d` runs the
  rotate / project / face-normal step on the GPU as a plain CUDA kernel
  (`project4d_quads_kernel`). Animation drivers detect "only 4D-projection
  params changed frame-to-frame" and re-project + refit the GAS+IAS
  in-place via `updateMesh4DProjection` instead of rebuilding the scene.
  Measured on `tesseract-sponge level=2`: ~30× setup, ~270× animation
  speed-up; equivalence with the CPU path is L∞ = 0/255 on static frames.
  See AD-19.
- **`--max-ray-depth N` CLI flag** (Sprint 18.5) — exposes the per-ray
  bounce/refraction recursion ceiling (1..8, default 5). Wired all the
  way through `RenderConfig` → JNI → `params.max_ray_depth` (the runtime
  path was already in place; 18.5 added the CLI surface and a regression
  test that asserts pixel-difference > epsilon across depths 2/4/8 on a
  glass-stack scene).
- **Render-health diagnostic** (Sprint 18.6) —
  `RenderHealth.check(pixels, w, h)` detects frames that are uniformly
  one colour (≥ 99 % of pixels within ε of a single RGB). On detection
  the CLI logs the offending invocation, refuses to write the PNG, and
  exits with status 2. Bypass with `--allow-uniform-render` for
  legitimate uniform scenes.

### Fixed
- **Multi-object 4D interactive hang** — pressing any 4D rotation key on a
  multi-object CPU-projected scene (e.g. mixed fractional+integer tesseract
  sponge) no longer hangs the application. A new `updateCpuTriangleMesh` JNI
  call rebuilds only the affected GAS in-place and re-links the IAS instance,
  bypassing `clearAllInstances()` + full scene rebuild which deadlocked the
  IAS rebuild path. The new `tryRotation4DCpuFastPath` mirrors the existing
  GPU fast path for all 4D-only CPU-projected scenes.
- **Edge rendering broken for tesseract/tesseract-sponge** — the CPU 4D fast
  path in `InteractiveEngine` was routing edge-rendering scenes to
  `TriangleMeshSceneBuilder` instead of `TesseractEdgeSceneBuilder`, causing
  cylinder edges to disappear. The fast path now correctly excludes scenes
  with `hasEdgeRendering`, allowing them to fall through to the normal scene
  classifier which selects the correct builder.

### Changed
- Sprint 18 architectural decisions documented as AD-17 through AD-20 in
  `docs/arc42/09-architectural-decisions.md`.
- User guide gains "Recursion Depth" and "Render Health Checks"
  subsections under OptiX Mode.
- Render-health diagnostic methodology moved to `debugging-rendering-bugs` skill.

## [0.5.7] - 2026-04-14

### Added
- **Scene graph with transform hierarchy and per-node material inheritance** — scenes can now be
  composed as a tree of `SceneNode`s, each carrying a local `Transform` (translation, rotation,
  scale) and an optional inherited `Material`. Transforms accumulate down the tree; materials
  propagate to descendants unless overridden. The flat `List[SceneObject]` API remains fully
  supported for backward compatibility; `Scene.root: Option[SceneNode]` enables the new tree API.
  `Vec3` gains `+` and `*` operators to support transform composition.
- **Engine trait composition** — `OptiXEngine` / `AnimatedOptiXEngine` class hierarchy replaced
  with a composable trait system: `RenderEngine` (base), `WithAnimation`, `WithPreview`,
  `WithVideoExport`. Concrete classes `InteractiveEngine`, `AnimationEngine`, `PreviewEngine`,
  and `VideoEngine` are composed from these traits, eliminating the need for deep inheritance.
- **DSL render settings** — all current render-quality settings are now expressible in the Scala
  DSL via `RenderSettings` (shadows, transparent shadows, antialiasing, AA depth/threshold) and
  `Camera.fov`. CLI flags override DSL values; DSL values override built-in defaults.
- **Video output via ffmpeg** — `--video output.mp4` (H.264/libx264) or `--video output.mkv`
  (HEVC/hevc_nvenc) encodes the frame sequence produced by `--frames` into a video file. Quality
  is controlled by `--video-quality` (QP, default 12 = master quality). Frame PNG files are
  deleted after encoding by default; `--keep-frames` retains them. ffmpeg availability and encoder
  support are validated at CLI parse time.

### Fixed
- **Menger sponge tunnel intrusion at level 2+** — faces with negative normals (-X,
  -Y, -Z) produced tunnel wall sub-faces whose normals pointed into the solid instead
  of into the tunnel. At deeper recursion levels this caused sub-tunnel geometry to be
  placed inside the parent tunnel, creating visible artifacts. Fixed by negating the
  rotation axes in `Face.rotatedSubFaces` for negative-normal parent faces.
- **Fractional sponge cross-fade making entire sponge transparent** — at fractional
  levels (e.g. 1.5), both the next-level mesh and the skin mesh were given reduced
  alpha, making the whole sponge see-through. Corrected so only the skin fades out
  while the next-level mesh remains fully opaque.

### Removed
- **LibGDX 3D rendering path** — the OpenGL/LibGDX 3D scene graph (`MengerEngine`,
  `InteractiveMengerEngine`, `AnimatedMengerEngine`, `ModelFactory`, `DragTracker`) has been
  removed. OptiX is now the sole renderer. LibGDX is retained as the windowing framework.
- **`--optix` CLI flag** — no longer needed; all rendering uses OptiX by default. The flag was
  previously required to access OptiX rendering; now `--objects` or `--scene` suffice.
- `proguard-base` dependency removed from build (was only needed by the LibGDX 3D path).

## [0.5.6] - 2026-04-01

### Added
- Project website at https://lilacashes.gitlab.io/menger/ — MkDocs site with render gallery and feedback links
- CUDA 13 parallel CI jobs (`Test:Full:Cuda13`, `Test:OptiXJni:Cuda13`) run alongside CUDA 12.8 jobs as `allow_failure` to catch version compatibility issues early
- AWS spot instance workflow polish: AMI IDs persisted in `scripts/ami-registry.tsv` (version-controlled, region-aware); `--ami-id` is now optional when an AMI exists for the active region; `--list-amis` subcommand shows all built AMIs; `--menger-branch` sets the git branch cloned and built on the instance; `--retrieve GLOB` retrieves artifacts from `~/GLOB` on the instance after `--command` completes; SSH polling now exits with a clear error and recovery instructions on timeout; `menger-app` is built via `sbt stage` and installed to `~/bin` during instance initialisation
- `nvtop` installed in AMI for GPU monitoring on spot instances
- `build-ami.sh` multi-region AMI distribution: `--copy <ami-id> --to-regions REGION[,...]` copies an existing AMI to additional regions; `--copy-to-regions` copies immediately after a fresh build; copied AMIs are registered in `ami-registry.tsv` automatically
- Cloud GPU development guide (`docs/guide/cloud.md`) covering full spot instance workflow: prerequisites, first-time setup, launching, headless renders, state management, spot termination protection, cost control, AMI registry management, and troubleshooting

## [0.5.5] - 2026-03-28

### Added
- **Soft shadows with area lights** — a new `AreaLight` type (disk emitter) casts soft shadows
  with visible penumbra. Shadow samples are configurable per-light (1–16, default 4). Available
  in the Scala DSL (`AreaLight(position, normal, radius, shadowSamples = 4)`) and via the CLI
  (`--light area:px,py,pz:nx,ny,nz:radius[:samples[:intensity[:color]]]`).
- **Parametric surfaces `f(u,v) → Vec3`** — arbitrary surfaces defined by a Scala function are
  CPU-tessellated into triangle meshes and rendered through the full OptiX pipeline. Supports
  open and closed (seam-welded) surfaces, configurable resolution (`uSteps`, `vSteps`), automatic
  normal computation from partial derivatives, and all existing material types including glass and
  PBR. Built-in example scenes: sphere, torus, wavy sheet, Möbius strip, Klein bottle.
  DSL: `ParametricSurface(f = (u, v) => Vec3(...), uRange, vRange, uSteps, vSteps, closedU, closedV)`.
- **Caustics for arbitrary geometry** — photon tracing now dispatches through the SBT to
  geometry-specific closest-hit programs, generalizing beyond sphere-only refraction. Parametric
  meshes (torus, Klein bottle, etc.) and any other refractive triangle geometry now produce
  physically correct caustics.

## [0.5.4] - 2026-03-24

### Added
- **Colored transparent shadows (Phase 2)** — shadow rays now accumulate attenuation
  multiplicatively through all transparent objects using anyhit programs. Phase 1 only captured
  the first transparent object; stacked or overlapping transparent objects now contribute their
  combined tint. Backward-compatible: when `--transparent-shadows` is off, anyhit programs
  return immediately, preserving Phase 1 behavior.
  - `accumulateShadowAttenuation()` device helper (screen-blend formula) in `helpers.cu`
  - `__anyhit__shadow()` with sphere EXIT-hit deduplication (hit_kind != 0 is ignored)
  - `__anyhit__triangle_shadow()` with back-face deduplication
  - `__anyhit__cylinder_shadow()`
  - `createHitgroupProgramGroupWithAH` / `createTriangleHitgroupProgramGroupWithAH` overloads
    in `OptiXContext` for hitgroup creation with all three program slots

### Fixed
- **Non-deterministic caustics reference images** — removed pixel-comparison reference images
  for `DSL_GlassSphere` and `DSL_CausticsDemo` integration tests, which use stochastic PPM
  caustics. These tests now run as smoke tests (render-without-crash) to avoid intermittent CI
  failures from photon-mapping randomness.

## [0.5.3] - 2026-03-16

### Added
- **Plane material presets** — `--plane-material <name>` applies a full PBR material preset
  (chrome, gold, glass, etc.) to the ground plane, enabling mirror floors, metallic surfaces,
  and other material-driven plane appearances. Mutually exclusive with `--plane-color`.
- **Material physical plausibility validation** — `Material.validate()` in the Scala DSL
  returns advisory warnings for physically implausible combinations (e.g., metallic material
  with IOR > 1.1, thin-film on metal, high roughness + high metallic).
- **MixedMetallicShowcase DSL example scene** — demonstrates partial metallic values
  (0.0, 0.25, 0.5, 0.75, 1.0) side-by-side to visualise the metallic–dielectric continuum.
- **Colored transparent shadows (Phase 1)** — transparent objects cast color-tinted shadows
  when `--transparent-shadows` is enabled. Shadow color is derived from the object's material
  color and opacity. Phase 1 supports single-object shadows; multi-object accumulation
  deferred to Phase 2.

### Fixed
- **SBT offset fix for triangle mesh rendering** — single-object triangle meshes now correctly
  use their own hitgroup records (primary + shadow) instead of accidentally hitting sphere
  hitgroups. Adds `sbt_base_offset` to `Params` struct, computed as
  `geometry_type * STRIDE_RAY_TYPES`.
- **Triangle/cylinder shadow payload encoding** — `__closesthit__triangle_shadow` and
  `__closesthit__cylinder_shadow` now correctly encode alpha using `__float_as_uint(alpha)`,
  matching the sphere shadow shader. Previously, triangle shadows used raw integer `1`
  (interpreted as ~0.0 float = no shadow) and cylinder shadows were a no-op (also no shadow).
  These latent bugs were exposed by the SBT offset fix routing shadow rays to the correct
  geometry-specific hitgroups.

### Removed
- Unused anyhit shadow programs and overloads from reverted colored shadow attempt (dead code
  cleanup)

## [0.5.2] - 2026-03-05

### Added
- **t-Parameter Animation System** — animate DSL scenes using a free parameter `t`
  - Animated scenes define `def scene(t: Float): Scene` instead of `val scene: Scene`
  - `--t <value>` CLI option: evaluate an animated scene at a fixed t value (freeze-frame)
  - `--frames N --start-t F --end-t F` CLI options: render multi-frame animations by sweeping t
  - `AnimatedOptiXEngine`: per-frame scene rebuild with full OptiX re-render
  - `LoadedScene` ADT (`Static` / `Animated`) for type-safe scene loading
  - `SceneLoader` auto-detects animated scenes via reflection (`def scene(Float)` vs `val scene`)
  - `SceneConverter` utility: reusable DSL-to-config conversion for both static and animated paths
  - `TAnimationConfig` with linear t interpolation across frame sequence
  - Example animated scenes: `OrbitingSphere` (sphere orbiting origin), `PulsingSponge` (varying fractal level)
  - CLI validation: `--t` mutually exclusive with `--start-t`/`--end-t`/`--frames`; both require `--scene` and `--optix`
  - 27 new unit tests (`TAnimationConfigSuite`, `TAnimationCLIOptionsSuite`, animated scene tests)
  - Integration tests for freeze-frame and multi-frame t-animation
- **Multiple Ground Planes** — scenes now support up to 4 independent planes (previously one fixed floor)
  - DSL `planes: List[Plane]` replaces `plane: Plane`; `Plane` objects carry position, normal, and color
  - JNI API updated to pass a plane array; OptiX shaders iterate all active planes per ray
- **Vec3 Rotation for All DSL Objects** — `rotation` field on all scene objects now takes `Vec3(xDeg, yDeg, zDeg)` (was `Float` for y-axis only; breaking change for `Sponge.rotation`)
- `SceneClassifier` object — shared scene classification and builder selection logic extracted from `OptiXEngine` and `AnimatedOptiXEngine`; 15 unit tests in `SceneClassifierSuite`
- `KeyRotation` trait — shared `factor` map and `angle()` calculation for `GdxKeyHandler` and `OptiXKeyHandler`
- `LibGDXConverters.toGdxButton` — `MouseButton` → LibGDX button code conversion moved from `GdxCameraHandler` extension method

### Fixed
- `ScreenshotFactory.sanitizePath` now preserves absolute paths — was incorrectly stripping the leading `/`, causing multi-frame animation frames to be saved to a relative `tmp/...` path instead of the specified `/tmp/...` absolute path
- **Emissive transparent triangle meshes** — `getTriangleMaterial` now extracts and passes `mat.emission` through the Fresnel blend path; was silently 0.0f regardless of material settings
- **Animated 4D edge scenes** — `AnimatedOptiXEngine` now correctly routes tesseract-edge scenes to `TesseractEdgeSceneBuilder` via shared `SceneClassifier`; was silently using `TriangleMeshSceneBuilder`
- `AnimatedOptiXEngine.render()` wraps `sceneFunction(t)` in `Try`; a throwing scene function now logs and skips the frame instead of crashing the application
- `Plastic` and `Matte` DSL material presets now delegate to `OptixMaterial.Plastic`/`OptixMaterial.Matte`; previously hardcoded inline (single source of truth completed)
- Named constants `THIN_FILM_COSINE_CLAMP_MIN`, `THIN_FILM_AIRY_DENOM_GUARD`, `CIE_Y_INTEGRAL_NORM` replace magic literals in `computeThinFilmReflectance`

### Changed
- `MockModelFactory` moved from `src/main/scala` (production code) to `src/test/scala` (test-only) — test doubles must not be compiled into the production artifact
- `computeEffectiveMaxInstances` extracted from three duplicate blocks in `OptiXEngine` to a private helper; `OptiXEngine` reduced from 488 to ~430 lines
- `computeEyeW` formula extracted to `CameraHandler` trait — shared by `GdxCameraHandler` and `OptiXCameraHandler`
- `DragTracker`: backing field renamed `_origin` → `dragOrigin` (Scala naming convention)

## [0.5.1] - 2026-02-24

### Added
- **Scala libGDX Wrapper** (`menger.gdx`) — all `var` and `null` for libGDX confined to a dedicated wrapper layer; non-wrapper Scala code uses `val` and `Option` throughout
  - `GdxRuntime` — lifecycle and exit; `KeyPressTracker` — Shift/Ctrl state; `DragTracker` — mouse drag delta; `OrbitCamera` — spherical camera orbit
  - `ModelFactory` — abstraction for LibGDX model creation with `LibGDXModelFactory` (production) and `MockModelFactory` (testing without LibGDX initialization)
  - Input handlers (`GdxKeyHandler`, `GdxCameraHandler`, `OptiXKeyHandler`, `OptiXCameraHandler`) rewritten to delegate all mutable state to the wrapper
  - `Builder` uses dependency-injected `ModelFactory` instead of direct `ModelBuilder` instantiation
  - 17 model caching tests in `ModelFactorySuite` running without LibGDX/LWJGL window creation
- **4D Framework Enhancements**
  - Shift+Scroll adjusts 4D projection distance (`eyeW`) interactively in OptiX mode
  - ESC resets the 4D view (rotation and projection) to its initial state without affecting the 3D camera
  - `--rotation-4d=XW,YW,ZW` CLI shorthand — specifies all three 4D rotation angles in a single option; mutually exclusive with `--rot-x-w`/`--rot-y-w`/`--rot-z-w`
- **DSL 4D Object Support** (Task 11.9) — Tesseract and TesseractSponge now available in DSL
  - `Tesseract` case class with projection, edge rendering, material, and position support
  - `TesseractSponge` case class with `VolumeRemoving` and `SurfaceSubdividing` types
  - `TesseractDemo` example scene demonstrating 4D object DSL usage
  - Full test coverage in `SceneObjectSuite`
- **Thin-Film Interference** — physically-based iridescent materials (soap bubbles, oil slicks)
  - Airy reflectance formula with 16-sample CIE 1931 XYZ spectral integration (380–780 nm)
  - `film-thickness=NM` CLI parameter on any object; applied per-wavelength for RGB iridescence
  - `material=film` preset: IOR 1.33, 500 nm (constructive interference at green), 20% transparency
  - Coated materials: combine any material with `film-thickness=NM` for oil-slick effects
  - `FilmSphere` DSL example scene: three spheres at 300 nm (violet), 500 nm (green), 700 nm (red)
  - 8 GPU integration tests in `FilmRenderSuite`; 8 CLI scenarios in `integration-tests.sh`

### Changed
- **Coverage baseline auto-update** (Task 11.8) — `.coverage_baseline` now updates after successful pre-push hook; baseline updated from 78.01% to 84.90%
- **USER_GUIDE modernization** (Task 11.10) — All deprecated `--object`, `--radius`, `--ior` examples replaced with `--objects 'type=...:key=value'` syntax; old flags marked as deprecated in reference documentation

### Removed
- **GdxTest window tests** (Task 11.7) — Removed 12 libGDX window tests from `GeometrySuite` that required display initialization (flaky in CI); kept two pure toString tests

### Fixed
- **TesseractSponge vertex validation** (Task 11.11) — Added test verifying that all `TesseractSponge2` (surface subdivision) vertices lie within the corresponding `TesseractSponge` (volume removal) region

## [0.5.0] - 2026-02-18

### Added
- **Scala DSL for Scene Description** - Type-safe scene definition language compiled with the project
  - Core DSL types: `Vec3`, `Color`, `Material` with intuitive constructors and implicit conversions
  - Scene objects: `Sphere`, `Cube`, `Sponge` (VolumeFilling, SurfaceUnfolding, CubeSponge)
  - Lights: `Directional` and `Point` lights with tuple position syntax
  - Camera: Position/lookAt configuration with tuple support
  - Plane: Ground plane with axis syntax (`Y at -2`) and solid/checkered colors
  - Caustics: High-quality photon mapping configuration (`Caustics.HighQuality`)
  - Material presets: Glass, Water, Diamond, Chrome, Gold, Copper with `.copy()` customization
  - Material factories: `Material.matte()`, `Material.plastic()`, `Material.metal()`, `Material.glass()`
  - Scene composition with compile-time type checking and IDE support
- **Scene Loader** - Dynamic scene loading via reflection
  - `--scene <classname>` CLI option for loading pre-compiled DSL scenes
  - SceneRegistry for short-name aliases (e.g., `--scene glass-sphere`)
  - Dual loading mechanism: registry lookup + reflection-based class loading
  - Clear error messages for missing scenes or invalid class names
- **9 Example Scenes** - Comprehensive DSL demonstrations
  - `SimpleScene` - Minimal single sphere example
  - `ThreeMaterials` - Glass, Chrome, Gold material showcase
  - `CausticsDemo` - High-quality caustics rendering with glass sphere
  - `CustomMaterials` - 5 custom materials using `.copy()` and factories
  - `ComplexLighting` - Multi-light setup (key, fill, rim, warm/cool accents)
  - `SpongeShowcase` - Three sponge types comparison
  - `MengerShowcase` - Classic Menger Sponge with three-point lighting
  - `GlassSphere` - Glass sphere with caustics on white floor
  - `ReusableComponents` - Pattern for importing common materials/lighting
- **Reusable Component Libraries** - Shared materials and lighting setups
  - `examples.dsl.common.Materials` - 20+ custom materials (TintedGlass, BrushedGold, RoseGold, Pearl, Obsidian, Terracotta, etc.)
  - `examples.dsl.common.Lighting` - 8 pre-configured lighting setups (ThreePointLighting, DramaticLighting, GoldenHourLighting, StudioLighting, RimLighting, ColoredAccentLighting, NightSceneLighting, SoftAmbientLighting)
  - Import and reuse across multiple scenes for consistency
- **DSL Integration Tests** - 164 comprehensive tests
  - PlaneSuite (22 tests) - Axis helpers, solid/checkered planes, validation
  - ExampleScenesSuite (10 tests) - All example scenes load via reflection
  - Material, Light, Camera, Scene, Caustics, Color, Vec3 test suites

### Changed
- Scene definition now supports compile-time type-safe DSL as alternative to CLI
- Example scenes moved from `examples/dsl/` to `menger-app/src/main/scala/examples/dsl/` for proper compilation
- DSL material presets (Glass, Water, Diamond, Chrome, Gold, Copper, Film, Parchment) now delegate to `OptixMaterial` — single source of truth for IOR/roughness/alpha values (M4)
- Redundant Float/Int/Double tuple overloads removed from `Directional`, `Point`, `Sphere`, `Cube`, `Sponge`, `Camera` companions; `Vec3` implicit conversions cover all cases (L15)
- `Plane.toPlaneColorSpec` uses pattern match instead of unsafe `Option.get` (L14)
- `setupShaderBindingTable` in `PipelineManager.cpp` decomposed into `createRaygenRecord`, `createMissRecords`, `createHitgroupRecords` helpers (L7)
- C++ magic numbers replaced with named constants: `NUM_PROGRAM_GROUPS`, `MAX_PHOTON_THREADS_PER_ROW`, `MIN_CONTINUATION_STACK_SIZE`, `DIELECTRIC_ALPHA`, `VERTEX_STRIDE_WITH_ALPHA` (L6, L8, L9, L10, L13)

### Fixed
- **Fractional sponge rendering** - Sponge geometry at fractional levels (e.g., 0.5, 1.5) now renders correctly
- **Duplicate sponge mesh logic** - `buildFractionalMesh` extracted to `FractionalLevelSponge` trait; `SpongeByVolume` and `SpongeBySurface` share the implementation (L12)
- **SceneRegistry short-name lookup** - `SceneIndex` forces eager initialization of all example scene objects at startup so registry names are reliably populated (L16)
- **Zero-vector division** - `normalize()` and `normalize3f()` in `VectorMath.h` now guard against zero-length input (M2)
- **setLights bounds** - `SceneParameters::setLights()` clamps count to `MAX_LIGHTS` and guards against null pointer (M3)
  - Added coverage alpha blending shader path: `vertex_alpha × diffuse + (1 − vertex_alpha) × continuation_ray`
  - Previously, partial alpha triggered the Fresnel refractive path, which had no effect on white/gray materials
  - Fixed z-fighting between skin and sponge meshes by expanding skin vertices 0.0003 world units outward along normals
  - Fix applied to all sponge types: `SpongeByVolume`, `SpongeBySurface`, and 4D sponge projections

## [0.4.3] - 2026-02-05

### Added
- **Sprint Planning Reorganization** - Restructured sprints 10-14 based on completed work
  - Sprint 10: Scala DSL for Scene Description (prioritized for better workflow)
  - Sprint 11: 4D Framework Enhancements (remaining UX features)
  - Sprint 12: Visual Quality & Materials (new sprint from TODO priorities)
  - Sprint 13: Object Animation Foundation
  - Sprint 14: Advanced Animation System
  - Documentation: SPRINT_REORGANIZATION_2026-02.md explains rationale

### Changed
- **Test Performance** - Parallelized integration tests for 1.73x speedup
  - Integration tests now run scenarios in parallel using xargs
  - Total integration test time reduced from ~45s to ~26s
  - All 27 scenarios still validate correctly
- **CLI Cleanup** - Removed legacy CLI options
  - Removed: `--radius`, `--ior`, `--scale`, `--center` (replaced by `--objects` syntax)
  - All examples and documentation updated to use modern `--objects` syntax
  - Backward compatibility: old options show migration message

### Fixed
- **Shadow Ray Direction** - Corrected directional light direction convention for shadow rays
  - Root cause: Commit bb92d30 incorrectly negated light direction, causing shadow rays to trace away from light sources
  - Shadow rays now correctly trace toward light sources, restoring shadow functionality
  - Established convention: `light.direction` points TO the light source (where light comes from)
  - All 26 shadow tests pass, all 87 integration tests pass
  - Documentation updated across Light.scala, USER_GUIDE.md, OptiXData.h, MengerCLIOptions.scala, helpers.cu
  - Reference images regenerated with corrected lighting
- **Parchment Material** - Corrected to be translucent instead of refractive
  - Changed `opacity` from 0.7 to 1.0 (fully opaque)
  - Parchment now behaves as a glowing matte material, not semi-transparent
- **CI Pipeline** - Fixed PushToGithub job missing git history
  - Added `GIT_DEPTH: 0` to fetch complete repository history
  - Ensures GitHub mirror has full commit history for proper release notes
- **Code Quality CI** - Resolved Docker API version mismatch
  - Updated code_quality job to use compatible Docker API version
  - CI pipeline now runs successfully without Docker socket errors

### Documentation
- **Release Checklist** - Added comprehensive release workflow documentation
  - Complete step-by-step guide from preparation to verification
  - Covers version management, testing, CI pipeline, and post-release checks
  - Documents TEST FAILURE PROTOCOL and common issues
  - Available via `/release-checklist` skill

## [0.4.2] - 2026-01-26

### Added
- **4D Menger Sponges (TesseractSponge)** - Fractal 4D geometry rendering
  - `--objects type=tesseract-sponge:level=N` for volume-based 4D Menger sponge (24 × 48^level faces)
  - `--objects type=tesseract-sponge-2:level=N` for surface-based 4D Menger sponge (24 × 16^level faces)
  - Fractional level support (e.g., `level=1.5` truncates to integer level)
  - Level parameter required and must be non-negative
  - Full 4D projection support (rot-xw, rot-yw, rot-zw, eye-w, screen-w)
  - Material support (glass, chrome, etc.) on projected sponge faces
  - Cylindrical edge rendering with `edge-material` and `edge-radius` parameters
- **Generalized 4D Projection Pipeline** - `Mesh4DProjection` class
  - Refactored `TesseractMesh` to accept any `Mesh4D` instance (not just Tesseract)
  - Backward-compatible `TesseractMesh` factory object preserves existing API
  - `TesseractSpongeMesh` and `TesseractSponge2Mesh` factories for convenient sponge creation
  - All 4D meshes now share the same projection, rotation, and translation logic
- **4D Sponge Type System** - Classification and validation
  - Extended `ObjectType` with `tesseract-sponge` and `tesseract-sponge-2`
  - New `ObjectType.is4DSponge()` helper method
  - Both types classified as hypercubes via `ObjectType.isHypercube()`
  - Validation enforces level requirement for 4D sponges in `ObjectSpec`
- **Performance Warnings** - Automatic threshold checks for high-level sponges
  - tesseract-sponge: warns at level ≥2 (55K faces), errors at level >4 (127M faces)
  - tesseract-sponge-2: warns at level ≥3 (98K faces), errors at level >5 (25M faces)
  - Estimated triangle counts logged to inform users of render complexity
  - Warnings are advisory only - no hard rejection of high levels
- **Edge Rendering for All 4D Types** - Generalized cylinder edge extraction
  - `TesseractEdgeSceneBuilder` supports tesseract, tesseract-sponge, tesseract-sponge-2
  - Dynamic edge extraction from any `Mesh4D` (not limited to 32 edges)
  - Edge count grows with sponge level (e.g., level 1 sponge has ~1,152 edges)
  - Instance budget calculation accounts for variable edge counts
- **Tesseract (4D Hypercube)** - Render 4D geometry projected to 3D via OptiX
  - `--objects type=tesseract` for 4D hypercube rendering (16 vertices, 24 faces projected to 3D)
  - 4D projection parameters: `eye-w=W`, `screen-w=W` (default: 3.0, 1.5)
  - 4D rotation parameters: `rot-xw=DEG`, `rot-yw=DEG`, `rot-zw=DEG` (default: 15°, 10°, 0°)
  - Full material support (glass, chrome, etc.) on tesseract faces
  - `TesseractMesh` class for 4D→3D projection with proper normals and UVs
- **Cylinder Primitive** - Custom OptiX primitive for edge rendering
  - Analytical ray-cylinder intersection in CUDA shader
  - Support for 32 cylindrical edges per tesseract
  - `addCylinderInstance()` API for cylinder instances with endpoints and radius
  - CLI: `--objects type=tesseract:edge-material=chrome:edge-radius=0.02`
- **Metallic Reflection on Cylinder Edges** - Single-bounce PBR reflection
  - Cylinder shader uses `handleMetallicOpaque()` for depth 0 metallic materials
  - Diffuse fallback for depth > 0 to prevent stack overflow
  - Stack size increased from 32KB to 48KB for metallic cylinder rendering
  - Chrome and copper edges show realistic mirror-like reflections
- **Interactive 4D Rotation** - Mouse-based manipulation of 4D objects
  - Left-drag: XW plane rotation (horizontal movement controls 4D rotation)
  - Right-drag: YW plane rotation (horizontal movement controls 4D rotation)
  - Middle-drag: ZW plane rotation (horizontal movement controls 4D rotation)
  - Vertical movement on all drags controls 3D camera pitch
  - Camera state preserved during 4D rotation (position, target, up vector)
- **Edge Material Properties** - Separate materials for tesseract edges
  - `edge-material=PRESET` for preset materials on edges (chrome, copper, glass, etc.)
  - `edge-color=#RRGGBB` for custom edge colors
  - `edge-emission=VALUE` for glowing edges (0.0-1.0)
  - `edge-radius=VALUE` for cylinder thickness (default: 0.02)
- **Emission Property** - Self-illuminating materials
  - Added `emission` field to Material case class (0.0-1.0)
  - Emissive materials glow without requiring light sources
  - Film and Parchment preset materials with emission values
- **Headless Rendering** - Batch processing without window display
  - `--headless` flag renders directly to file without displaying window
  - Invisible window creation using LibGDX's `setInitialVisible(false)`
  - Requires `--save-name` to be specified
  - Useful for CI/CD, batch processing, and remote servers
- **Scene Builder Architecture** - Strategy pattern for object type handling
  - `SceneBuilder` trait with validate/buildScene/calculateInstanceCount methods
  - `SphereSceneBuilder` for pure sphere scenes
  - `TriangleMeshSceneBuilder` for cubes and sponges
  - `CubeSpongeSceneBuilder` for GPU-instanced cube sponges
  - `TesseractEdgeSceneBuilder` for tesseracts with cylindrical edges
  - Automatic builder selection based on object types
  - Validation prevents incompatible object combinations
- **Input Abstraction Layer** - Clean separation of LibGDX and rendering logic
  - `InputEvent` ADT for key/mouse events (KeyPress, KeyRelease, MouseDrag)
  - `InputHandler` trait for processing input events
  - `GdxKeyHandler` and `OptiXKeyHandler` for specific handling
  - `GdxCameraHandler` and `OptiXCameraHandler` for camera manipulation
  - `LibGDXInputAdapter` bridges LibGDX callbacks to event system
  - Zero LibGDX dependencies in camera/key handler logic
- **User Guide** - Comprehensive documentation (1630 lines)
  - Quick start guide with installation and first render
  - Basic usage: spheres, cubes, sponges with materials
  - 4D visualization guide: tesseracts, rotation, projection
  - Headless rendering for batch processing
  - Advanced topics: multiple objects, custom lighting, performance
  - Examples gallery with render commands
- **Projection4DSpec** - 4D projection parameter encapsulation
  - Separate case class for 4D-specific parameters (eyeW, screenW, rotations)
  - Default values defined in companion object
  - Used by ObjectSpec for tesseract configuration

### Changed
- **MeshFactory** - Added 4D sponge cases
  - `tesseract-sponge` and `tesseract-sponge-2` now supported in `MeshFactory.create()`
  - Both use 4D projection parameters from `ObjectSpec.projection4D`
- **OptiX Engine** - Integrated performance warnings
  - `warnIfHighLevel()` now handles 4D sponge types with triangle estimates
- **CLI Help** - Updated `--objects` description
  - Clarified level requirement for 4D sponges: `level=L (required)`
  - Generalized 4D parameter descriptions
- OptiX pipeline now includes cylinder custom primitive hit groups
- Stack size increased from 32KB to 48KB for metallic cylinder shaders
- Shader file renamed from `sphere_combined.cu` to `optix_shaders.cu`
- Input handling refactored to use event-based architecture instead of controller pattern
- Camera manipulation extracted to separate handler classes
- Scene building logic extracted from OptiXEngine to dedicated builder classes
- Material extraction logic moved to `MaterialExtractor` utility
- Texture loading logic moved to `TextureManager` utility

### Technical Details
- **Architecture**: All 4D meshes (Tesseract, TesseractSponge, TesseractSponge2) implement `Mesh4D` trait
- **Projection**: 4D faces → 3D quads → 2 triangles per quad
- **Edge Extraction**: Canonical ordering deduplicates edges from quad faces
- **Backward Compatibility**: Existing `TesseractMesh` usage unaffected

### Fixed
- Screenshot vertical flip bug - images now saved with correct orientation
  - Added `flipVertically()` method in ScreenshotFactory
  - OpenGL framebuffer (bottom-left origin) correctly converted to PNG (top-left origin)
- Infinite pipeline rebuild loop for tesseract edge rendering
  - Fixed by properly tracking pipeline state in TesseractEdgeSceneBuilder
- Cylinder module cleanup causing double-free crash
  - Fixed GAS buffer management for cylinder primitives
- Crash when rotating tesseract with chrome edges
  - Resolved by implementing single-bounce reflection strategy

### Tests
- All 1159 tests passing (394 in menger-app, 765 in optix-jni + C++)
- Test coverage: 82.04% (up from 78.01%)
- 51 integration tests passing (basic objects, multi-object, materials, tesseract, headless)
- New test suites:
  - `TesseractMeshSuite` - 18 tests for 4D→3D projection
  - `TesseractIntegrationSuite` - 25 tests for tesseract rendering pipeline
  - `CylinderSuite` - 34 tests for cylinder primitive and intersection
  - `Camera4DRotationSuite` - 21 tests for interactive 4D rotation
  - `InputEventSuite` - 11 tests for input event system
  - `KeyHandlerSuite` - 12 tests for key event handling
  - `CameraHandlerSuite` - 6 tests for camera manipulation

## [0.4.1] - 2026-01-12

### Added
- **Material System** - PBR-based material properties for realistic rendering
  - Material case class with baseColor, metallic, roughness, IOR, alpha
  - Material presets: glass, water, diamond, chrome, gold, copper, metal, plastic, matte
  - Per-object material assignment via CLI: `--objects type=sphere:material=glass`
  - Per-object color override: `--objects type=cube:material=chrome:color=#FF0000`
- **UV Coordinates** - Texture mapping foundation
  - 8-float vertex format: position (3) + normal (3) + UV (2)
  - UV generation for cube and sponge meshes
  - Box mapping for procedural UV assignment
- **Texture Support** - Image-based surface coloring
  - `TextureLoader` utility for PNG/JPEG loading
  - `--texture-dir` CLI option for texture search path
  - Per-object texture assignment: `--objects type=cube:texture=checker.png`
  - Texture sampling in shaders with bilinear filtering
  - Per-instance texture indices in IAS mode
- **Multi-Project Build Structure** - Reorganized into modular subprojects
  - Separate modules: common, mengerApp, native
  - Improved build isolation and dependency management
- **Comprehensive Error Handling** - Robust error reporting and validation
  - Specific exception types: InvalidMaterialException, TextureLoadException, InvalidObjectSpecException
  - Detailed error context with actionable messages
  - Material and object specification validation
- **Test Coverage Improvements** - ~170 new tests for robustness
  - Property-based tests for animation parameters
  - Edge case tests for CLI parsing and object specifications
  - Coverage protection with ratchet mechanism (75.87% threshold)
- **Manual Test Script** - Comprehensive visual regression testing tool
  - Tests materials, textures, multi-object scenes, shadows, and reflections
- **Strategic Debug Logging** - Configurable diagnostic output for troubleshooting

### Changed
- OptiX shaders extended with texture sampling functions
- Vertex format updated from 6 to 8 floats per vertex
- Removed legacy `--object` CLI option (superseded by `--objects`)
- Reduced cognitive complexity across shaders and parsing code
- Extracted shared helper functions and regex patterns to common module

### Fixed
- Reflection formula in `traceReflectedRay` for accurate glass rendering
- Working directory for `sbt run` now correctly uses project root
- Screenshot path handling in ScreenshotFactory

## [0.4.0] - 2026-01-05

### Added
- **Instance Acceleration Structure (IAS)** - Multi-object rendering foundation
  - `addSphereInstance()` API for adding sphere instances with position and material
  - `addTriangleMeshInstance()` API for triangle mesh instances with transforms
  - Per-instance 4x3 transform matrices and material properties
  - GAS registry for geometry type management
  - `optixGetInstanceId()` for per-instance material lookup in shaders
- **Multi-Object CLI** - `--objects` parameter with keyword=value format
  - Support for sphere, cube, sponge-volume, sponge-surface, and cube-sponge types
  - Per-object position, size, color, IOR, and level parameters
  - Example: `--objects type=sphere:pos=-1,0,0:color=#FF0000`
- **Cube-Based Sponge (GPU Instancing)** - Memory-efficient sponge rendering
  - `CubeSpongeGenerator` generates instance transforms instead of merged geometry
  - One base cube mesh shared by all instances (up to 3.2M instances at level 5)
  - 40-80x memory reduction vs. merged mesh approach
  - CLI: `--objects type=cube-sponge:pos=0,0,0:level=2:color=#00FF00`
  - Configurable instance limit via `--max-instances` (default: 64)
- **Rendering Tests for IAS** - 18 tests including repeated render stress tests
- **Integration Tests** - Multi-object and triangle mesh rendering validation
- **Shadow Rays for Triangle Meshes** - Triangle meshes now cast shadows correctly
  - Shadow rays trace against IAS handle in multi-object mode
  - Shadow rays trace against GAS handle in single-object mode
  - Un-ignored test "cast shadows on the plane" now passes

### Fixed
- **CUDA error 700 in IAS mode** - Fixed use-after-free bug in GAS buffer management
  - IAS GAS buffers now managed separately from BufferManager
  - Multiple renders with IAS now work correctly
- **Triangle mesh shadows** - Triangle meshes now cast shadows on plane and other objects
  - params.handle correctly set to GAS/IAS handle depending on mode
  - Shadow ray shader works with both sphere and triangle geometry

## [0.3.9] - 2025-12-01

### Added
- **Triangle Mesh Support** - OptiX can now render triangle meshes (foundation for future geometry)
  - `setTriangleMesh()` API for uploading vertex/index buffers to GPU
  - Per-vertex normals for correct shading
  - Separate hit group programs for spheres and triangles
- **Cube Primitive** - First triangle mesh object, rendered via OptiX
  - 12 triangles with outward-facing normals
  - Demonstrates triangle intersection and shading

### Changed
- OptiX pipeline now supports both sphere and triangle geometry types
- SBT (Shader Binding Table) extended for multiple geometry hit groups

## [0.3.8] - 2025-11-28

### Added
- **Caustics Rendering** (experimental, deferred) - Progressive Photon Mapping groundwork
  - Architecture in place but algorithm issues identified
  - Enable with `--caustics` (may not produce visible results yet)

### Changed
- Refactored C++ architecture: decomposed OptiXWrapper into BufferManager and CausticsRenderer
- Enabled parallel test execution and enforced code quality tools across all subprojects

## [0.3.7] - 2025-11-21

### Added
- **Unified Color Type** - New `menger.common.Color` class for consistent color handling
  - RGBA components (0.0-1.0 range) with validation
  - `Color.fromRGB()`, `Color.fromRGBA()`, `Color.fromHex()` factory methods
  - `toRGBArray` and `toRGBAArray` methods for JNI conversion
- **Custom Plane Colors** - Configure plane colors via `--plane-color` flag
  - Solid color: `--plane-color #RRGGBB`
  - Checkered pattern: `--plane-color RRGGBB:RRGGBB`
- **Color Conversion Utilities** - Extension methods for LibGDX/common Color interop
  - `toCommonColor` extension on LibGDX Color
  - `toGdxColor` extension on menger.common.Color

### Changed
- Light colors now use `Color` type instead of `Vector[3]` for consistency
- `setSphereColor` float overloads are now private; use `Color` API instead

## [0.3.6] - 2025-11-20

### Added
- **Multiple Light Sources** - Configure up to 8 lights via `--light` flag
  - Format: `--light <type>:x,y,z[:intensity[:color]]`
  - Supports directional and point lights
  - Example: `--light directional:1,1,-1:2.0:ffffff --light point:0,5,0:3.0:ff0000`
- **Shadow Rendering** - Realistic hard shadows with `--shadows` flag
  - Transparent objects cast lighter shadows based on material alpha
  - Glass casts light shadows, opaque objects cast dark shadows

## [0.3.5] - 2025-11-17

### Added
- **Fresnel Reflection** - Realistic glass rendering with reflection and refraction blending

### Fixed
- Glass rendering now works correctly (previously showed only refraction)
- Improved transparency rendering accuracy for semi-transparent materials

## [0.3.4] - 2025-11-02

### Added
- **OptiX GPU Ray Tracing** - Hardware-accelerated sphere rendering with `--optix` flag
  - Configure sphere radius with `--sphere-radius <value>`
  - Save screenshots with `--save-name <filename>`
  - Auto-exit after timeout with `--timeout <seconds>`

## [0.3.3] - 2025-10-26

### Added
- OptiX ray tracing support for realistic lighting and materials

## [0.3.2] - 2025-10-23

### Added
- GPU rendering now optional (build works without CUDA/OptiX)

## [0.3.1] - 2025-10-21

### Added
- `--log-level` option to control logging verbosity (ERROR, WARN, INFO, DEBUG)
- `--fps-log-interval` option to control frequency of FPS logging

## [0.3.0] - 2025-10-06

### Added
- Remote GPU development support with AWS spot instances


## [0.2.9] - 2025-10-05

### Added
- Level animation support for all fractal sponge types via `--animate frames=N:level=start-end`
- Overlay rendering mode with `--face-color` and `--line-color` options for wireframe on transparent
  faces
- BlendingAttribute support in material builder for proper alpha transparency
- Comprehensive unit tests for level animation with single and chained animation specifications
- Validation to prevent parameters from being specified both as CLI option and in animation spec
- Validation to prevent conflicting color option combinations
- Documentation for animation parameters including level, rotation, 4D rotation, and projection
  settings
- Documentation for overlay mode with face and line color options
- Strict code quality enforcement with wartremover errors for Var, While, AsInstanceOf, IsInstanceOf,
  Throw
- Strict null checking with scalafix (noNulls = true)
- Compiler flag -Wunused:imports for continuous import validation
- @SuppressWarnings annotations for necessary vars in LibGDX integration and performance-critical code

### Fixed
- Refactored AnimationSpecification to eliminate code duplication in interpolation logic
- Fixed PushToGithub CI job by adding branch fetch before checkout
- Replaced mutable var with functional alternatives (AtomicBoolean, AtomicReference) where possible
- Replaced null checks with Option wrapper for LibGDX compatibility
- Removed 7 unused imports across the codebase
- Fixed scaladoc warnings by filtering out plugin options

### Upgraded
- sbt-scalafix 0.11.1 → 0.14.3
- logback-classic 1.5.18 → 1.5.19 (fixes CVE-2025-11226 security vulnerability)
- scalamock 7.4.1 → 7.5.0

## [0.2.8] - 2025-10-04

### Added
- Scalafix integration for code quality and automated refactoring
- Fractional level support for SpongeBySurface with smooth alpha transitions
- FractionalLevelSponge trait to eliminate code duplication between sponge implementations
- Fractional level support for TesseractSponge and TesseractSponge2 via FractionalRotatedProjection
  wrapper

### Fixed
- Path traversal vulnerability in screenshot filename handling with comprehensive test coverage
- Improved timing precision by replacing System.currentTimeMillis with System.nanoTime
- Made getIntegerModel a lazy val to prevent repeated sponge instantiation in render loop
- Corrected alpha calculation for fractional level sponges to properly transition from full opacity 
  to transparency
- Eliminated code duplication by moving createMaterialWithAlpha to FractionalLevelSponge companion 
  object

### Upgraded
- Updated dependencies: scala-logging 3.9.6, sbt-native-packager 1.11.3, sbt-scoverage 2.3.1, 
  sbt-jupiter-interface 0.11.3

## [0.2.7] - 2025-09-15

### Added
- `--color` option to set the color of the rendered object

### Upgraded
- Scala to 3.7.3
- sbt to 1.11.5

## [0.2.6] - 2025-08-12

### Added
- replaced LibGDX's `Vector4` and `Matrix4` with `Vector[4]` and `Matrix[4]` for future 
  extensibility

### Upgraded
- sbt to 1.11.4

## [0.2.5] - 2025-03-27

### Added
- script parameter animations
- Use named tuples throughout the code

### Upgraded
- Scala to 3.7.2
- sbt to 1.10.11
- Scalamock to 7.4.0

## [0.2.4] - 2025-03-20

### Added
- Clean up code by replacing Tuples with explicit classes

### Upgraded
- ScalaMock to 7.2.0, fixing resulting errors in tests

## [0.2.3] - 2025-03-17

### Added
- Visualize a four-dimensional Menger Sponge analog generated by subdividing each face into 16
  subfaces

### Upgraded
- Scala to 3.6.4
- Scalatest to 3.2.19

## [0.2.2] - 2024-10-30

### Upgraded
- Scala to 3.5.2
- Scalatest to 3.2.18
- Scallop to 5.1.0

## [0.2.1] - 2024-03-26

### Added
- Visualize a four-dimensional Menger Sponge analog generated by subdividing a Tesseract into 48 
  smaller Tesseracts
- Changelog with retroactive entries for previous versions

## [0.2.0] - 2024-03-19

### Added
- Visualize a Tesseract 
- Interactively rotate and change projection distance of the Tesseract

## [0.1.0] - 2024-02-20

### Initial Release
- Visualize Menger Sponge generated by subdividing a cube into 20 smaller cubes
- Visualize Menger Sponge generated by subdividing a face into 12 smaller faces


[0.7.5]: https://gitlab.com/lilacashes/menger/-/compare/0.7.4...0.7.5
[0.7.6]: https://gitlab.com/lilacashes/menger/-/compare/0.7.5...0.7.6
[0.7.4]: https://gitlab.com/lilacashes/menger/-/compare/0.7.3...0.7.4
[0.7.3]: https://gitlab.com/lilacashes/menger/-/compare/0.7.2...0.7.3
[0.7.2]: https://gitlab.com/lilacashes/menger/-/compare/v0.7.1...0.7.2
[0.7.1]: https://gitlab.com/lilacashes/menger/-/compare/v0.7.0...0.7.1
[0.7.0]: https://gitlab.com/lilacashes/menger/-/compare/v0.6.2...0.7.0
[0.6.1]: https://gitlab.com/lilacashes/menger/-/compare/0.6.0...0.6.1
[0.6.0]: https://gitlab.com/lilacashes/menger/-/compare/0.5.8...0.6.0
[0.5.8]: https://gitlab.com/lilacashes/menger/-/compare/0.5.7...0.5.8
[0.5.7]: https://gitlab.com/lilacashes/menger/-/compare/0.5.6...0.5.7
[0.5.6]: https://gitlab.com/lilacashes/menger/-/compare/0.5.5...0.5.6
[0.5.5]: https://gitlab.com/lilacashes/menger/-/compare/0.5.4...0.5.5
[0.5.4]: https://gitlab.com/lilacashes/menger/-/compare/0.5.3...0.5.4
[0.5.3]: https://gitlab.com/lilacashes/menger/-/compare/0.5.2...0.5.3
[0.5.2]: https://gitlab.com/lilacashes/menger/-/compare/0.5.1...0.5.2
[0.5.1]: https://gitlab.com/lilacashes/menger/-/compare/0.5.0...0.5.1
[0.5.0]: https://gitlab.com/lilacashes/menger/-/compare/0.4.3...0.5.0
[0.4.3]: https://gitlab.com/lilacashes/menger/-/compare/0.4.2...0.4.3
[0.4.2]: https://gitlab.com/lilacashes/menger/-/compare/0.4.1...0.4.2
[0.4.1]: https://gitlab.com/lilacashes/menger/-/compare/0.4.0...0.4.1
[0.4.0]: https://gitlab.com/lilacashes/menger/-/compare/0.3.9...0.4.0
[0.3.9]: https://gitlab.com/lilacashes/menger/-/compare/0.3.8...0.3.9
[0.3.8]: https://gitlab.com/lilacashes/menger/-/compare/0.3.7...0.3.8
[0.3.7]: https://gitlab.com/lilacashes/menger/-/compare/0.3.6...0.3.7
[0.3.6]: https://gitlab.com/lilacashes/menger/-/compare/0.3.5...0.3.6
[0.3.5]: https://gitlab.com/lilacashes/menger/-/compare/0.3.4...0.3.5
[0.3.4]: https://gitlab.com/lilacashes/menger/-/compare/0.3.3...0.3.4
[0.3.3]: https://gitlab.com/lilacashes/menger/-/compare/0.3.2...0.3.3
[0.3.2]: https://gitlab.com/lilacashes/menger/-/compare/0.3.1...0.3.2
[0.3.1]: https://gitlab.com/lilacashes/menger/-/compare/0.3.0...0.3.1
[0.3.0]: https://gitlab.com/lilacashes/menger/-/compare/0.2.9...0.3.0
[0.2.9]: https://gitlab.com/lilacashes/menger/-/compare/0.2.8...0.2.9
[0.2.8]: https://gitlab.com/lilacashes/menger/-/compare/0.2.7...0.2.8
[0.2.7]: https://gitlab.com/lilacashes/menger/-/compare/0.2.6...0.2.7
[0.2.6]: https://gitlab.com/lilacashes/menger/-/compare/0.2.5...0.2.6
[0.2.5]: https://gitlab.com/lilacashes/menger/-/compare/0.2.4...0.2.5
[0.2.4]: https://gitlab.com/lilacashes/menger/-/compare/0.2.3...0.2.4
[0.2.3]: https://gitlab.com/lilacashes/menger/-/compare/0.2.2...0.2.3
[0.2.2]: https://gitlab.com/lilacashes/menger/-/compare/0.2.1...0.2.2
[0.2.1]: https://gitlab.com/lilacashes/menger/-/compare/0.2.0...0.2.1
[0.2.0]: https://gitlab.com/lilacashes/menger/-/compare/0.1.0...0.2.0
[0.1.0]: https://gitlab.com/lilacashes/menger/-/commit/f90eee11
