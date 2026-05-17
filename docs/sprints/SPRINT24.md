# Sprint 24: Video Backgrounds

**Sprint:** 24 - Video Backgrounds
**Status:** Not Started
**Estimate:** ~15 hours
**Branch:** `feature/sprint-24`
**Dependencies:** Sprint 22 (env map DSL wiring, per-frame texture swap infrastructure)

---

## Goal

Allow animated `.mp4` (or `.mov`) videos as environment map backgrounds. Each animation
frame samples the correct video frame from the file, enabling backgrounds that change over
time — e.g., a rotating sky, fire, or crowd — synchronized with animated 3D/4D fractals.

DSL-only scope. No CLI options needed.

## Success Criteria

- [ ] `envMapVideo = Some(EnvMapVideo("background.mp4", fps = 25))` in DSL plays video as background
- [ ] Video frame selection correctly synchronized to animation t and fps
- [ ] Video frames decode as HDR-compatible float textures (or byte with tone mapping)
- [ ] Per-frame GPU texture swap is performant (no full reupload pipeline stall)
- [ ] Memory bounded: GPU holds at most N decoded frames at a time (ring buffer)
- [ ] All tests pass

---

## Tasks

### Task 24.1: ffmpeg/libav CMake Dependency

**Estimate:** 3h

Add `libavcodec` / `libavformat` / `libswscale` (ffmpeg libraries) as a CMake dependency
in `optix-jni/src/main/native/CMakeLists.txt`.

**Approach:** `find_package(FFmpeg COMPONENTS avcodec avformat avutil swscale)` with
fallback to system packages (`libavcodec-dev` on Ubuntu). Do NOT use FetchContent —
ffmpeg is large and complex; system install is the right approach.

**Minimum required versions:** libavcodec ≥ 58 (ffmpeg 4.x, widely available on Ubuntu 20.04+)

**CMake guard:** wrap in `if(FFMPEG_FOUND)` — if not found, video background feature is
disabled at build time, and DSL emits a clear error at runtime.

**Implementation:**
1. Add `find_package` to CMakeLists.txt
2. Create `VideoLoader.h` / `VideoLoader.cpp` in `optix-jni/src/main/native/` —
   C++ class wrapping libav for sequential frame decode
3. JNI binding: `VideoLoader` exposed via existing JNI boundary pattern

---

### Task 24.2: Frame Decoder (C++ VideoLoader)

**Estimate:** 4h
**Depends on:** 24.1

`VideoLoader` decodes video frames on demand and caches a sliding window.

**API:**
```cpp
class VideoLoader {
public:
    VideoLoader(const char* path);
    ~VideoLoader();  // releases all libav contexts

    int width() const;
    int height() const;
    double duration() const;    // seconds
    int frameCount() const;
    double nativeFps() const;

    // Decode frame at timestamp (seconds). Returns RGBA float pixels.
    // Cached: repeated calls for same timestamp are free.
    const float* frameAt(double timestamp);  // width*height*4 floats

    // Preload next N frames starting at timestamp (background prefetch hint)
    void prefetch(double timestamp, int nFrames);
};
```

**Implementation:**
- `avformat_open_input` → `avcodec_find_decoder` → `avcodec_open2` for H.264/VP9/etc.
- `av_seek_frame` for random access (needed if animation replays or jumps)
- `sws_scale` to convert decoded YUV → RGBA float (or RGBA uint8 + Scala-side conversion)
- Frame cache: `std::unordered_map<int, std::vector<float>>` keyed by frame index,
  max `MAX_CACHED_FRAMES = 8` (ring eviction, oldest frame evicted)

**HDR video:** Most `.mp4` files are SDR (8-bit). HDR `.mp4` (HDR10 / HLG) exists but
requires tone curve handling. For Sprint 24: support SDR video; document HDR video as
future work. Tone mapping from Sprint 22 applies to video-sourced backgrounds the same way.

---

### Task 24.3: JNI Binding + Scala VideoLoader

**Estimate:** 2h
**Depends on:** 24.2

Expose `VideoLoader` through JNI:
- `VideoLoader.scala` in `optix-jni/src/main/scala/menger/optix/` wrapping native calls
- Native methods: `openVideo(path)`, `videoWidth()`, `videoHeight()`, `frameCount()`,
  `getFrameAt(timestamp): Array[Float]`, `closeVideo()`
- `OptiXVideoApi.scala` trait following existing `OptiXTextureApi` pattern

---

### Task 24.4: Per-Frame Texture Swap in Animation Loop

**Estimate:** 3h
**Depends on:** 24.3, Sprint 22 (env map wiring)

Extend `WithAnimation.scala` to swap the GPU env map texture each frame for video backgrounds.

**Frame selection:**
```scala
val videoTimestamp: Double = t * scene.envMapVideo.get.totalDuration  // or: frameIndex / fps
val frameData: Array[Float] = videoLoader.getFrameAt(videoTimestamp)
renderer.uploadTexture("__video_bg__", frameData, videoLoader.width, videoLoader.height)
renderer.setEnvironmentMap(textureIndex)
```

**Performance constraint:** Uploading a 4K frame (4096×2048 × 4 × 4 bytes = 128 MB) every
frame is a bottleneck. Mitigations:
- Upload on a background CUDA stream while previous frame renders
- Default video background to 2K resolution max (user can override)
- `prefetch()` hint to VideoLoader for next 2 frames

**Ring buffer on GPU:** Alternative — upload all needed frames at startup if `frameCount ≤ 30`
and total GPU memory budget allows. Use `params.env_map_frame_index` to select in shader.
Decision: implement streaming (per-frame upload) first; ring buffer is an optimization.

---

### Task 24.5: DSL Integration

**Estimate:** 1h
**Depends on:** 24.3

```scala
case class EnvMapVideo(
  path: String,
  fps: Double = 25.0,           // Playback fps (can differ from render fps)
  loop: Boolean = true,          // Loop video when animation t > video duration
  startOffset: Double = 0.0,     // Start playback at this timestamp (seconds)
)

// Scene.scala — add alongside envMap (Sprint 22)
case class Scene(
  ...,
  envMap: Option[String] = None,           // Static HDR (Sprint 22)
  envMapVideo: Option[EnvMapVideo] = None, // Animated video (Sprint 24) — mutually exclusive with envMap
  ibl: Option[IBL] = None,                 // Sprint 23
  toneMapping: ToneMapping = ToneMapping.Reinhard(),
)
```

`envMap` and `envMapVideo` are mutually exclusive; `SceneConfigurator` emits error if both set.

---

### Task 24.6: Documentation

**Estimate:** 2h

- User guide: "Video Backgrounds" section
  - Supported formats (H.264 .mp4, VP9 .webm — whatever libav decodes on target platform)
  - `fps` vs render fps: how they interact, frame selection math
  - Performance: frame size recommendations, prefetch behavior
  - HDR video: note that HDR10/HLG is not supported; use HDR .hdr frame sequences if needed
- Example scene: 4D sponge animating in front of time-lapse cloud video
- Sprint retrospective, CHANGELOG.md

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 24.1 | ffmpeg CMake dep + VideoLoader skeleton | 3h | — |
| 24.2 | VideoLoader C++ frame decoder | 4h | 24.1 |
| 24.3 | JNI binding + Scala VideoLoader | 2h | 24.2 |
| 24.4 | Per-frame texture swap in animation loop | 3h | 24.3, Sprint 22 |
| 24.5 | DSL `EnvMapVideo` type + `Scene.envMapVideo` | 1h | 24.3 |
| 24.6 | Documentation | 2h | All |
| **Total** | | **~15h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] Test video background committed as small `.mp4` for integration tests

---

## Notes

### ffmpeg Licensing

libav (ffmpeg) is LGPL 2.1+. Linking as a shared library (`.so`) keeps Menger's license
unaffected. Do NOT statically link ffmpeg — that would impose GPL on the binary.
CMakeLists.txt must link `${FFMPEG_LIBRARIES}` as shared.

### Supported Codecs

libav decodes whatever the system ffmpeg installation supports. H.264 (most .mp4) and VP9
(most .webm) are universally available. HEVC (H.265) requires system libx265. Menger makes
no codec promises — decoding falls back to libav error if codec unavailable.

### Alternative: Pre-Rendered Frame Sequence

Users who prefer not to install libav (e.g., minimal CI environments) can pre-render video
to `.hdr` frame sequences using ffmpeg CLI. Sprint 22's env map supports single `.hdr` files.
A frame sequence approach (Sprint 22 `scene(t)` function switching `envMap` per t-range) is
fully functional without Sprint 24. Sprint 24 adds convenience, not new capability.

### Memory Budget

A 2K equirectangular frame (2048×1024 float4) = 32 MB GPU. An 8-frame ring buffer = 256 MB.
4K = 4× that. Warn at runtime if ring buffer exceeds 512 MB VRAM.

### Deferred

- **HDR video (HDR10/HLG)** — requires tone curve (PQ/HLG → linear float). Later sprint.
- **GPU ring buffer pre-load** — optimization; streaming per-frame is the Sprint 24 approach
- **Audio synchronization** — out of scope (Menger renders images, not video with audio)
