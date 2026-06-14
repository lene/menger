# Sprint 27: Video Textures and 360 Video Backgrounds

**Sprint:** 27 - Video Textures and 360 Video Backgrounds
**Status:** In Progress
**Estimate:** ~32 hours
**Branch:** `feature/sprint-27`
**Dependencies:** Sprint 21.6 (image texture support), Sprint 22 (env map DSL wiring),
Sprint 24-26 repository/module split
**Implementation guide:** [SPRINT27_IMPLEMENTATION_PLAN.md](SPRINT27_IMPLEMENTATION_PLAN.md)

---

## Goal

Allow animated `.mp4` (or `.mov`) videos to drive render textures. The first target is
normal rectangular video mapped onto existing textured objects. Once that path works,
extend the same decoder and per-frame upload machinery to equirectangular 360-degree
video environment backgrounds, including IBL when the scene enables `ibl`.

DSL-only scope. No CLI options needed.

## Success Criteria

- [x] `videoTexture = Some(VideoTexture("texture.mp4"))` in DSL plays a rectangular
      video on any object that already supports image textures
- [x] `VideoTexture` and `EnvMapVideo` share playback config for time mapping,
      start offset, optional fps override, and repeat mode
- [x] Video frame selection is deterministic from animation `t`, source timestamps,
      optional `fpsOverride`, and repeat mode
- [x] Render `t` exceeding source duration is handled by repeat mode:
      loop, freeze, or ping-pong
- [x] Video frames decode to RGBA8 textures for SDR video
- [x] Per-frame object texture updates use stable GPU texture slots and avoid full
      scene rebuild when only the video frame changes
- [x] Memory is bounded: decoded CPU frames and uploaded GPU texture state are limited
      by explicit cache/slot ownership
- [x] `envMapVideo = Some(EnvMapVideo("background_360.mp4"))` plays an equirectangular
      360-degree video as the environment background after video textures are complete
- [x] `EnvMapVideo` updates IBL lighting when the scene enables `ibl`
- [x] FFmpeg/libav is installed and exercised in CI; video tests must not silently skip
- [x] All tests pass

---

## Tasks

### Task 27.0: Sprint Plan and Dependency Preflight

**Estimate:** 1h
**Status:** Complete

Keep this sprint plan aligned with the current three-module architecture and verify the
native dependency surface before implementation.

**Implementation:**
1. [x] Mark Sprint 27 as in progress in this file.
2. [x] Confirm all native video work is scoped to `menger-geometry`, not the split
   `optix-jni` repository.
3. [x] Verify local and CI availability for `libavcodec-dev`, `libavformat-dev`,
   `libavutil-dev`, and `libswscale-dev`.
4. [x] Verify whether the current `optix-jni` dependency exposes an update-in-place texture
   API. If it does not, add that generic renderer API before task 27.6.
5. [x] Harden standalone `optix-jni` CI/release policy so follow-up texture API work can
   be gated on every branch push/PR and published through Maven Central.
6. [x] Harden standalone `menger-common` CI/release policy with branch/PR gates,
   coverage ratchet, Scaladoc, MiMa, Java consumer smoke, `main` release gate,
   tag publication, and Maven Central post-publish smoke.

**Preflight result:**
- Native video decode belongs in `menger-geometry`; `optix-jni` remains the generic
  renderer/texture API dependency.
- Local libav dev headers are installed and visible to `pkg-config`:
  `libavcodec 60.31.102`, `libavformat 60.16.100`, `libavutil 58.29.100`,
  `libswscale 7.5.100`.
- CI jobs use apt-capable Ubuntu/CUDA images; task 27.1 installs the same libav dev
  packages before native build/test jobs.
- The consumed `io.github.lene:optix-jni:0.1.2` artifact does not expose
  `updateTexture`; task 27.6 remains blocked until a new `optix-jni` release with
  update-in-place texture API is published and this repo bumps to it.

---

### Task 27.1: ffmpeg/libav CMake Dependency in `menger-geometry`

**Estimate:** 2h
**Status:** Complete

Add `libavcodec` / `libavformat` / `libavutil` / `libswscale` as a CMake dependency
in `menger-geometry/src/main/native/CMakeLists.txt`.

**Approach:** `find_package(FFmpeg REQUIRED COMPONENTS avcodec avformat avutil swscale)`
with fallback to system packages (`libavcodec-dev` on Ubuntu). Do NOT use FetchContent -
ffmpeg is large and complex; system install is the right approach.

**Minimum required versions:** libavcodec ≥ 58 (ffmpeg 4.x, widely available on Ubuntu 20.04+)

**Dependency policy:** once video support lands, FFmpeg/libav is required in CI and video
tests must run. Missing native dependencies should fail the build with a clear CMake
error, not silently disable the feature.

**Implementation:**
1. [x] Add `find_package` to CMakeLists.txt
2. [x] Create `VideoLoader.h` / `VideoLoader.cpp` in `menger-geometry/src/main/native/` —
   C++ class wrapping libav for sequential frame decode
3. [x] JNI binding: `VideoLoader` exposed via existing JNI boundary pattern
4. [x] Ensure packaging/docs identify the runtime shared-library dependency

**Result:** CMake now requires and dynamically links `avcodec`, `avformat`, `avutil`,
and `swscale`; CI installs the corresponding Ubuntu dev packages before native builds.
The OptiX integration job timeout is raised to 30 minutes because the package build plus
integration suite exceeded the previous 20-minute limit after adding the dependency.
`VideoLoader` opens videos and exposes metadata through JNI/Scala. RGBA frame decode and
cache behavior remain Task 27.2.

---

### Task 27.2: Frame Decoder (C++ VideoLoader)

**Estimate:** 4h
**Depends on:** 27.1
**Status:** Complete

`VideoLoader` decodes video frames on demand and caches a sliding window.

**API:**
```cpp
struct VideoFrame {
    int width;
    int height;
    double timestampSeconds;
    std::vector<uint8_t> rgba;
};

class VideoLoader {
public:
    VideoLoader(const char* path);
    ~VideoLoader();

    int width() const;
    int height() const;
    double durationSeconds() const;
    int frameCount() const;
    double nativeFps() const;

    // Decode RGBA8 frame at timestamp (seconds). Cached.
    VideoFrame frameAt(double timestampSeconds);

    void prefetch(double timestampSeconds, int nFrames);
};
```

**Implementation:** `avformat_open_input` → `avcodec_find_decoder` → `sws_scale` YUV→RGBA8.
Frame cache: `std::unordered_map<int, std::vector<uint8_t>>`, max 8 frames (ring eviction).
Videos without alpha decode with `alpha = 255`. Frame orientation must match existing
`TextureLoader` image texture orientation. Invalid, unsupported, zero-frame, or
zero-duration videos fail fast during open/decode.

**Result:** `VideoLoader.frameAt` decodes SDR video frames to top-to-bottom RGBA8 through
libavcodec/libswscale, clamps timestamps to the first/last decodable frames, and keeps a
bounded eight-frame decoded cache. A tiny deterministic PNG-in-MOV fixture verifies exact
RGBA bytes through the Scala/JNI path.

---

### Task 27.3: JNI Binding + Scala VideoLoader

**Estimate:** 2h
**Depends on:** 27.2
**Status:** Complete

- `VideoLoader.scala` in `menger-geometry`
- Native methods: `openVideo(path)`, `videoWidth()`, `videoHeight()`, `frameCount()`,
  `videoDurationSeconds()`, `nativeFps()`, `getFrameAt(timestamp): Array[Byte]`,
  `closeVideo()`
- `MengerVideoApi` trait following existing JNI pattern
- Scala loader implements `AutoCloseable`; every engine path that opens videos closes
  them during renderer disposal or failure cleanup

**Result:** `VideoLoader` now extends a `MengerVideoApi` native-boundary trait,
exposes metadata, RGBA frame decode, prefetch, and idempotent close through Scala/JNI,
and rejects access after close. No renderer/engine path opens videos yet; later tasks
must use the `AutoCloseable` loader ownership point when wiring video textures.

---

### Task 27.4: DSL `VideoTexture` Type for Rectangular Object Textures

**Estimate:** 2h
**Depends on:** 27.3
**Status:** Complete

Add a DSL type for animated rectangular videos used as object textures. This is the first
user-facing video feature and must land before 360-degree backgrounds.

```scala
enum VideoTimeMapping:
  case AnimationRange
  case TSeconds
  case TProgress

enum VideoRepeat:
  case Loop
  case Freeze
  case PingPong

case class VideoPlayback(
  timeMapping: VideoTimeMapping = VideoTimeMapping.AnimationRange,
  repeat: VideoRepeat = VideoRepeat.Loop,
  startOffset: Double = 0.0,
  fpsOverride: Option[Double] = None,
)

case class VideoTexture(
  path: String,
  playback: VideoPlayback = VideoPlayback(),
)
// SceneObject.videoTexture: Option[VideoTexture]
```

Validation:
- `texture` and `videoTexture` are mutually exclusive for the same object.
- Normal-map and roughness-map videos are out of scope for this sprint.
- `VideoTexture` is supported for every object that already accepts image textures.
- `fpsOverride`, when set, must be positive. Without it, source timestamps/native fps
  drive frame selection.

**Result:** Added the shared `VideoPlayback`, `VideoTimeMapping`, `VideoRepeat`, and
`VideoTexture` DSL model, exported it through `menger.dsl`, and propagated
`SceneObject.videoTexture` into `ObjectSpec` for every DSL object that already accepts
image textures. `ObjectSpec` rejects simultaneous `texture` and `videoTexture`, and
focused tests cover default playback, FPS validation, conversion, and object support.

---

### Task 27.5: Static Initial-Frame Video Texture

**Estimate:** 2h
**Depends on:** 27.4
**Status:** Complete

Decode the initial playback frame of a rectangular video and upload it through the
existing texture path for objects that already support image textures. This proves the
decoder, JNI, Scala wrapper, and OptiX texture upload path before animation is introduced.

**Validation:** A DSL scene with a video-textured plane or cube renders the initial
playback frame as a stable still texture.

**Result:** `TextureManager` now decodes a `VideoTexture` playback start frame through
`menger.geometry.VideoLoader`, uploads the RGBA bytes through the existing texture upload
path, and binds the resulting stable texture slot through the same image-texture wiring
used by static textures. Added `examples.dsl.VideoTextureCube`, integration/manual test
entries, a committed reference image, and focused tests for decoded initial/start-offset
frames.

---

### Task 27.6: Animated Rectangular Video Texture Updates

**Estimate:** 4h
**Depends on:** 27.5
**Status:** Complete

Extend `WithAnimation.scala` to update object textures each frame when the only texture
change is a video frame.

Frame selection:
```scala
val rawTime = VideoPlaybackTime.resolve(
  playback = videoTexture.playback,
  renderT = t,
  animationRange = currentAnimationRange,
  durationSeconds = videoLoader.durationSeconds,
)
val sampleTime = VideoRepeatPolicy.resolve(
  rawTime,
  videoLoader.durationSeconds,
  videoTexture.playback.repeat,
)
val frameData = videoLoader.getFrameAt(sampleTime)
renderer.updateTexture(videoTextureSlot, frameData, videoLoader.width, videoLoader.height)
```

Use the existing scene-builder texture assignment path to bind each video-textured
instance to one stable texture slot. Avoid a full scene rebuild when only the video
frame changes. Do not call `uploadTexture` every frame unless it is changed to update
existing native texture memory instead of allocating a new texture.

Frame-stability rule: each rendered output frame samples one video frame. Accumulation
samples for that output frame must reuse the same video frame. Interactive video frame
advancement resets accumulation before sampling the next video frame.

**Result:** Root `build.sbt` now consumes `optix-jni:0.1.3` with update-in-place texture
support. `WithAnimation` records video texture slots from the existing scene-builder
upload path, updates those slots with `renderer.updateTexture` for stable video-textured
specs, and skips full scene rebuilds when only the sampled video frame changes. Added
deterministic playback timing/repeat helpers plus unit tests for `VideoTimeMapping`,
`VideoRepeat`, `fpsOverride`, and `t` values beyond the source duration.

---

### Task 27.7: DSL `EnvMapVideo` Type for 360-Degree Video Backgrounds

**Estimate:** 2h
**Depends on:** 27.6
**Status:** Complete

```scala
case class EnvMapVideo(
  path: String,
  playback: VideoPlayback = VideoPlayback(),
)
// Scene.envMapVideo: Option[EnvMapVideo] — mutually exclusive with envMap
```

`SceneConverter` emits an error if both `envMap` and `envMapVideo` are set. `EnvMapVideo`
expects equirectangular 2:1 360-degree video; ordinary rectangular videos remain object
textures. The same `VideoPlayback` rules apply to object videos and environment videos.

**Result:** Added `EnvMapVideo` to the shared video model with the same `VideoPlayback`
configuration used by `VideoTexture`, exported it through `menger.dsl`, added
`Scene.envMapVideo`, and propagated it through `SceneConverter.SceneConfigs` and
`EnvironmentConfig`. `SceneConverter` now rejects scenes that set both `envMap` and
`envMapVideo`. Focused tests cover the model defaults/keying/path validation,
conversion, IBL config propagation for `envMapVideo`, and the mutual-exclusion error.

---

### Task 27.8: Animated 360-Degree Environment Backgrounds

**Estimate:** 3h
**Depends on:** 27.7
**Status:** Complete

Reuse `VideoLoader` and the stable texture-slot update path to update the renderer
environment map from an equirectangular 360-degree video.

When the scene enables `ibl`, `EnvMapVideo` also updates the IBL source for the same
sampled video frame. When `ibl` is absent, `EnvMapVideo` is visible-background only.

**Validation:** A DSL animation with `envMapVideo` renders a changing background without
changing scene geometry, and an `ibl` scene shows lighting changes from the same video
frames.

**Result:** `WithAnimation` now uploads one stable texture slot for `envMapVideo`, validates
that the decoded video dimensions are equirectangular 2:1, updates that slot in place
before each animation frame, binds the slot with `setEnvironmentMap`, and reapplies IBL
against the same slot when `ibl` is enabled. Static DSL rendering through `InteractiveEngine`
uses the deterministic initial frame as the environment map. Added the deterministic
`video/two-frame-equirect-rgba.mov` 4x2 fixture, generated with:
`ffmpeg -y -f lavfi -i color=c=red:s=4x2:r=2:d=0.5 -f lavfi -i color=c=blue:s=4x2:r=2:d=0.5 -filter_complex '[0:v][1:v]concat=n=2:v=1:a=0,format=rgba' -c:v qtrle -pix_fmt argb menger-geometry/src/test/resources/video/two-frame-equirect-rgba.mov`.
Focused tests cover fixture decode, 2:1 validation, env-video slot reuse eligibility, and
video-source detection.

---

### Task 27.9: Memory Bound and Texture Update Performance

**Estimate:** 3h
**Depends on:** 27.6, 27.8
**Status:** Complete

Keep decoded CPU frames and uploaded GPU texture state bounded.

**Renderer contract:**
- Allocate one stable GPU texture slot per active video source.
- Update that slot in place as frames change.
- Share one decoder/cache/texture slot for identical video source + playback config.
- Verify the implementation uses the update-in-place renderer API added or confirmed
  before task 27.6. Do not emulate video playback by allocating one texture per frame.

**CPU cache:** start with a small decoded-frame cache, max 8 frames per active video
source. Expand only if profiling shows avoidable seek/decode stalls.

**Result:** Animated video playback now owns reusable per-source decoder/cache state with an
8-frame LRU decoded-frame cache. Object video texture rebuilds use a TextureManager slot provider
to reuse active texture indices instead of re-uploading the same video source, including repeated
loads during mixed-scene builds. Object and environment video updates both use `updateTexture` on
stable texture slots, and animation disposal closes active decoders.

---

### Task 27.10: Documentation and Examples

**Estimate:** 2h
**Status:** Complete

- User guide: "Video Textures" section — rectangular video textures, supported formats,
  source fps vs `fpsOverride`, playback time mapping, repeat modes, and performance
  recommendations
- User guide: "360 Video Backgrounds" section — equirectangular video requirement,
  difference from normal object videos, IBL behavior, and accumulation behavior
- Example scene: a plane or cube with a normal rectangular video texture
- Example scene: 4D sponge in front of a 360-degree time-lapse video
- arc42 update: native libav dependency, stable texture update API, video-memory risk,
  and dynamic env-map/IBL behavior
- CHANGELOG.md entry

**Result:** Added user-guide and DSL-reference coverage for `VideoTexture`,
`EnvMapVideo`, `VideoPlayback`, time mapping, repeat modes, `fpsOverride`, 2:1
equirectangular requirements, IBL behavior, accumulation behavior, and performance
recommendations. Added `examples.dsl.EnvMapVideoSponge`, integration/manual coverage for
one rectangular video texture and one env-map video, fixture-generation documentation,
arc42 updates for native libav/stable texture slots/video memory risk, and a changelog
entry. CI libav package installs now avoid recommended distro CUDA packages so GPU jobs
keep using the driver-mounted `libcuda.so.1`.

---

### Task 27.11: Fix M-sceneb-validate-bypass

**Estimate:** 4h
**Depends on:** none (pure Scala refactor)
**Status:** Complete

Fix `CODE_IMPROVEMENTS.md` `M-sceneb-validate-bypass` before or alongside video texture
work so invalid scene input cannot reach GPU instance allocation.

**Changes:**
- Ensure `buildSceneFromConfigs` validates before build for all scene paths.
- Unify or consistently bridge `SceneBuilder.validate` and `buildScene` error types.

**Result:** Added `SceneBuilder.validateAndBuild` to bridge validation failures into the
existing `Try[Unit]` build contract via `ValidationException`. Routed config-based,
mixed-scene, rebuild, interactive 4D tracked, and animated 4D tracked builds through the
validated path before renderer allocation. Added engine-level regression coverage for
invalid triangle-mesh configs and mixed-scene groups, and removed the resolved
`M-sceneb-validate-bypass` item from `CODE_IMPROVEMENTS.md`.

---

### Task 27.12: Fix M-instanceid-raw-int — Opaque InstanceId type

**Estimate:** 4h
**Depends on:** none (pure Scala refactor)
**Status:** Complete

Replace raw `Int` instance IDs from native `add*Instance` methods with an opaque type to
enforce -1-to-failure translation at the API boundary instead of each caller.

**Changes:**
- Introduce `opaque type InstanceId = Int` (or a value class wrapper) in the engine/scene layer
- Return `Option[InstanceId]` or `Try[InstanceId]` from public `add*` wrappers
- Remove scattered `-1` checks across `CubeSpongeSceneBuilder`, `ConeSceneBuilder`,
  `Menger4DSceneBuilder`, etc.
- Update `update*4DProjection` callers to use typed IDs - eliminates mid-frame
  `IllegalArgumentException` risk

See `CODE_IMPROVEMENTS.md` `M-instanceid-raw-int`.

**Result:** Added an opaque scene-layer `InstanceId` and centralized native `-1`
translation through `SceneBuilder.requireInstanceId`. Updated primitive, mesh, cube-sponge,
edge-rendered tesseract, and direct 4D builders to fail scene construction instead of
logging and continuing after failed instance allocation. Interactive direct-4D fast-path
caches now store typed instance IDs and unwrap only at native `update*4DProjection` calls;
full rebuilds reuse tracked builders instead of synthesizing IDs from sequential indices.
Added regression coverage for native Menger4D allocation failure and removed the resolved
`M-instanceid-raw-int` item from `CODE_IMPROVEMENTS.md`.

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 27.0 | Sprint plan and dependency preflight | 1h |
| 27.1 | ffmpeg CMake dep in menger-geometry + VideoLoader skeleton | 2h |
| 27.2 | VideoLoader C++ frame decoder | 4h |
| 27.3 | JNI binding + Scala VideoLoader | 2h |
| 27.4 | DSL VideoTexture type for rectangular object textures | 2h |
| 27.5 | Static initial-frame video texture | 2h |
| 27.6 | Animated rectangular video texture updates | 4h |
| 27.7 | DSL EnvMapVideo type for 360-degree backgrounds | 2h |
| 27.8 | Animated 360-degree environment backgrounds | 3h |
| 27.9 | Memory bound and texture update performance | 3h |
| 27.10 | Documentation and examples | 2h |
| 27.11 | Fix M-sceneb-validate-bypass | 4h |
| 27.12 | Fix M-instanceid-raw-int (opaque InstanceId type) | 4h |
| **Total** | | **~32h** |

---

## Definition of Done

- [x] All success criteria met
- [x] `./.git_hooks/pre-push 2>&1 | tee /tmp/pre-push.log` passes
- [x] CHANGELOG.md updated
- [x] Small rectangular test video committed for video texture integration tests
- [x] Small equirectangular/360 test video committed for env-map video integration tests
- [x] Test videos are deterministic, tiny, SDR RGBA-visible patterns generated from
      repo-owned source or documented commands
- [x] `scripts/integration-tests.sh` covers at least one video texture and one env-map video
- [x] `scripts/manual-test.sh` covers at least one video texture and one env-map video
- [x] Unit tests cover `VideoTimeMapping`, `VideoRepeat`, `fpsOverride`, and `t`
      exceeding source duration
- [x] CI installs FFmpeg/libav dev packages and runs the video integration scenarios
- [x] arc42 sections 9, 10, and 11 updated for video decode, texture update, and risks

---

## Notes

### Playback Semantics

`VideoTexture` and `EnvMapVideo` use the same `VideoPlayback` config.

`VideoTimeMapping.AnimationRange` is the default. It maps the current animation range
to one video duration. If no animation range exists, such as a single `--t` render, it
falls back to `TProgress`.

`VideoTimeMapping.TSeconds` treats `t` as seconds. `VideoTimeMapping.TProgress` treats
`t = 0..1` as one full video duration. `startOffset` is added before repeat behavior is
applied.

`VideoRepeat.Loop` wraps by positive modulo. `VideoRepeat.Freeze` clamps to the first or
last decodable frame. `VideoRepeat.PingPong` reflects across the duration repeatedly.
These rules apply when render `t` exceeds video length and when effective time is
negative.

Source timestamps/native fps are used by default. `fpsOverride` is optional and replaces
source timing only when explicitly set.

### Texture Update Contract

Animated video must update existing GPU texture memory. A renderer implementation that
allocates a new texture for every frame does not satisfy this sprint because it makes GPU
memory growth unbounded.

### ffmpeg Licensing

libav (ffmpeg) is LGPL 2.1+. Link as shared library only. Do NOT statically link.

### SDR Only

Sprint 27 supports SDR video (8-bit). HDR10/HLG is future work.

### Ordering

Normal rectangular animated video textures are the first deliverable. 360-degree video
backgrounds are intentionally second because they reuse the same decoder, timestamp,
cache, and per-frame upload machinery, but target the environment-map slot instead of
object image textures.
