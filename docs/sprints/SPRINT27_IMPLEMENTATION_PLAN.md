# Sprint 27 Implementation Plan

This document is the implementation guide for Sprint 27. It complements
`SPRINT27.md` by spelling out the execution order, code boundaries, API changes,
validation rules, and test expectations so another engineer can implement the sprint
without re-deciding the design.

Sprint 27 adds deterministic video-driven render textures in two stages:

1. Rectangular SDR videos as object image textures.
2. Equirectangular 2:1 SDR videos as environment backgrounds, including IBL when
   enabled.

The implementation must keep video frame changes out of normal scene geometry rebuilds
and must keep GPU texture memory bounded.

## Current Facts

- Branch: `feature/sprint-27`.
- Current `optix-jni` dependency: `io.github.lene:optix-jni:0.1.2`.
- Current `optix-jni` texture API exposes `uploadTexture`, `uploadTextureFromFile`,
  `setImageTexture`, `setEnvironmentMap`, and `releaseTextures`.
- Current `optix-jni` texture API does not expose update-in-place texture replacement.
- Static image textures are currently loaded once by `TextureManager` during scene
  build.
- `WithAnimation` currently rebuilds scene geometry unless it can use the existing 4D
  projection fast path.
- `TAnimationConfig` provides `startT`, `endT`, `frames`, and `tForFrame(frameIndex)`.

## Phase 0: `optix-jni` Texture Update API

Animated video requires a stable GPU texture slot that can receive new RGBA8 bytes. Do
this in the split `optix-jni` repository before implementing animated updates in this
repository.

Add this public Scala API to `OptiXTextureApi`:

```scala
def updateTexture(
  textureIndex: Int,
  imageData: Array[Byte],
  width: Int,
  height: Int
): Unit
```

Requirements:

- Validate `textureIndex >= 0`.
- Validate `imageData != null`.
- Validate `width > 0` and `height > 0`.
- Validate `imageData.length == width * height * 4`.
- Native implementation updates existing texture memory for that slot.
- Native implementation must not allocate a new texture slot per frame.
- Add Scala and native tests proving repeated updates reuse the same texture index.

After publishing the new `optix-jni` version:

- Do not infer the version number. Ask the maintainer for the new version.
- Bump `optixJniDependency` in root `build.sbt`.
- Keep all generic texture update behavior in `optix-jni`; keep video decode in
  `menger-geometry`.

## Phase 1: Quality Blockers

Implement these before adding video behavior.

### Scene Validation Bypass

Fix `M-sceneb-validate-bypass`.

Add a shared helper in `BaseEngine`:

```scala
private def validateThenBuild(
  builder: SceneBuilder,
  specs: List[ObjectSpec],
  renderer: OptiXRenderer,
  maxInstances: Int
): Try[Unit]
```

Use the helper for every `buildScene` call in:

- `buildSceneFromSpecs`
- `buildSceneFromConfigs`
- mixed-scene branches
- rebuild paths where validation can run before construction

Validation failures must return `Failure(ValidationException(...))`. Avoid throwing
through `.get` in scene construction paths.

### Instance ID Boundary

Fix `M-instanceid-raw-int`.

Introduce a typed instance ID wrapper in the scene or engine layer. Convert native `-1`
failures once, immediately after each native `add*Instance` call.

Rules:

- Scene builders should use typed IDs or `Try[InstanceId]`/`Option[InstanceId]`.
- Do not pass raw negative IDs into update methods.
- 4D fast-path caches should store typed IDs, or convert to raw `Int` only at the final
  renderer API boundary.

## Phase 2: Native Video Decoder in `menger-geometry`

Add libav/FFmpeg support in `menger-geometry`, not in `optix-jni`.

Native paths:

- `menger-geometry/src/main/native/CMakeLists.txt`
- `menger-geometry/src/main/native/include/VideoLoader.h`
- `menger-geometry/src/main/native/VideoLoader.cpp`
- `menger-geometry/src/main/native/MengerJNIBindings.cpp`

CMake requirements:

- Require `avcodec`, `avformat`, `avutil`, and `swscale`.
- Missing libav dependencies fail the build clearly.
- Link dynamically only.

Native decoder contract:

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

    VideoFrame frameAt(double timestampSeconds);
    void prefetch(double timestampSeconds, int nFrames);
};
```

Rules:

- Decode SDR video to RGBA8.
- Videos without alpha use `alpha = 255`.
- Match existing `TextureLoader` image orientation.
- Fail fast on invalid path, unsupported codec, no video stream, zero frames, or zero
  duration.
- Cache at most 8 decoded frames per loader.
- Use RAII for libav objects and frame buffers.

Expose a Scala `VideoLoader` in `menger-geometry` with `AutoCloseable`.

## Phase 3: DSL and Playback Model

Add video model types in a neutral package such as `menger.video`, then re-export them
from `menger.dsl`.

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

case class EnvMapVideo(
  path: String,
  playback: VideoPlayback = VideoPlayback(),
)
```

Add:

- `SceneObject.videoTexture: Option[VideoTexture]`
- `Scene.envMapVideo: Option[EnvMapVideo]`

Validation:

- `texture` and `videoTexture` are mutually exclusive.
- `envMap` and `envMapVideo` are mutually exclusive.
- `fpsOverride`, if present, must be positive.
- Video normal maps and roughness maps are out of scope.
- Video values must survive `SceneNode` flattening without becoming decoded frame data.

## Phase 4: Playback Time Resolver

Create a pure playback resolver with unit tests.

Input:

- `VideoPlayback`
- current render `t`
- optional animation range
- video duration
- native fps or source timestamp metadata

Behavior:

- `AnimationRange`: map `startT..endT` to one video duration.
- If there is no usable animation range, fall back to `TProgress`.
- `TSeconds`: `rawTime = startOffset + t`.
- `TProgress`: `rawTime = startOffset + t * duration`.
- `Loop`: positive modulo over duration.
- `Freeze`: clamp to first or last frame.
- `PingPong`: reflect repeatedly across duration.
- Negative effective time follows the selected repeat mode.
- `fpsOverride` quantizes sampled time when set.
- Source timestamps/native fps drive decode when `fpsOverride` is absent.

## Phase 5: Engine-Level Video Runtime

Add an engine-owned video runtime. Do not put long-lived video state inside a single
scene-builder call.

Responsibilities:

- Resolve video paths relative to `textureDir`.
- Open and close `VideoLoader`s.
- Share one decoder/cache/texture slot for identical absolute path plus playback config.
- Allocate one stable GPU texture slot per active video source.
- Upload the initial frame once.
- Update the existing slot with `renderer.updateTexture`.
- Close unused loaders on engine disposal.
- Keep decoded CPU frames and GPU texture slots bounded.

Suggested concepts:

```scala
final case class VideoSourceKey(
  absolutePath: String,
  playback: VideoPlayback
)

final case class ActiveVideoTexture(
  key: VideoSourceKey,
  textureIndex: Int,
  loader: VideoLoader
)
```

The runtime should update video slots before `renderScene`, never during native
accumulation.

## Phase 6: Rectangular Video Textures

Extend texture binding so static and video texture sources can coexist.

Implementation direction:

- Keep static image loading in `TextureManager`.
- Add video slot resolution through the new video runtime.
- Builders bind `videoTexture` objects to the stable video texture slot with
  `setImageTexture`.
- A video frame update must not change normal `ObjectSpec` equality.
- If only the sampled frame changes, do not rebuild the scene.
- If video path or playback config changes, rebuild and rebind.

Support every object type that already accepts image textures.

## Phase 7: Animation and Preview Integration

Update `WithAnimation`:

- Build the first frame normally.
- Before each render, resolve current video sample time from `t`.
- Update all active video slots.
- Then render.
- Accumulation must sample one stable video frame per output frame.

Update `WithPreview`:

- Use preview `t` as the playback source.
- Step/play behavior updates video frames deterministically.
- Do not introduce wall-clock playback in Sprint 27.

Static non-preview scenes sample the deterministic initial playback frame only.

## Phase 8: 360 Environment Video and IBL

Add `envMapVideo` to `SceneConverter.SceneConfigs`.

Rules:

- Open and validate equirectangular 2:1 dimensions.
- Bind the env video texture slot with `setEnvironmentMap`.
- If `ibl` is configured, the same updated slot drives IBL.
- If `ibl` is absent, video is visible-background only.
- Update the env-map slot before each render frame.

## Phase 9: Test Fixtures and Integration Coverage

Add tiny deterministic test videos:

- Rectangular video: small SDR clip with obvious per-frame color or pattern changes.
- Equirectangular video: 2:1 SDR clip with obvious left/right/top/bottom pattern
  changes.

Fixtures must be repo-owned or generated by documented commands.

Unit tests:

- Playback mapping for all three `VideoTimeMapping` values.
- Repeat behavior for `t == duration`, `t > duration`, negative time, and ping-pong.
- `fpsOverride` quantization.
- DSL validation for texture/video conflicts.
- DSL validation for `envMap`/`envMapVideo` conflicts.
- `BaseEngine` validation helper covers mixed and non-mixed scene paths.
- `InstanceId` wrapper rejects `-1` and preserves valid IDs.

Native/JNI tests:

- Decode known tiny video dimensions.
- Read duration, frame count, and native fps.
- Verify first and middle frame colors.
- Invalid input fails clearly.
- Close is idempotent and releases native state.

Renderer/API tests:

- `optix-jni` update-in-place API validates dimensions and RGBA byte length.
- Repeated frame updates reuse one texture slot.

Integration tests:

- Object video texture.
- Object video texture with `t > duration`.
- Env-map video.
- Env-map video with IBL.

Manual tests:

- One object video texture scene.
- One env-map video scene.

Video-input tests must not silently skip when FFmpeg/libav is missing.

## Phase 10: CI, Docs, and Definition of Done

CI:

- Ensure GPU test jobs install or image-bake libav dev/runtime packages.
- `Test:Full` and `Test:OptiXIntegration` must exercise video support.
- Existing video-output tests may still skip on missing `ffmpeg` executable.
- New video-input tests must not skip.

Docs:

- Update user guide with video texture syntax, playback modes, repeat modes, and 360
  video requirements.
- Update `CHANGELOG.md`.
- Update arc42:
  - Section 9: decision for native libav decode and stable GPU texture slots.
  - Section 10: video playback quality and performance scenarios.
  - Section 11: risks for JNI decoder leaks, GPU texture memory, and color/format
    confusion.

Final verification:

```bash
./.git_hooks/pre-push 2>&1 | tee /tmp/pre-push.log
```

If any rendering output changes intentionally, update reference images in the same
commit or an immediately following test-only commit on the same branch.

## Out of Scope

- CLI flags for video textures.
- Wall-clock video playback.
- HDR10/HLG video.
- Video normal maps.
- Video roughness maps.
- Audio.
