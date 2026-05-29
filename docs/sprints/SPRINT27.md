# Sprint 27: Video Backgrounds

**Sprint:** 27 - Video Backgrounds
**Status:** Not Started
**Estimate:** ~15 hours
**Branch:** `feature/sprint-27`
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

### Task 27.1: ffmpeg/libav CMake Dependency

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

### Task 27.2: Frame Decoder (C++ VideoLoader)

**Estimate:** 4h
**Depends on:** 27.1

`VideoLoader` decodes video frames on demand and caches a sliding window.

**API:**
```cpp
class VideoLoader {
public:
    VideoLoader(const char* path);
    ~VideoLoader();

    int width() const;
    int height() const;
    double duration() const;
    int frameCount() const;
    double nativeFps() const;

    // Decode frame at timestamp (seconds). Cached.
    const float* frameAt(double timestamp);  // width*height*4 floats

    void prefetch(double timestamp, int nFrames);
};
```

**Implementation:** `avformat_open_input` → `avcodec_find_decoder` → `sws_scale` YUV→RGBA.
Frame cache: `std::unordered_map<int, std::vector<float>>`, max 8 frames (ring eviction).

---

### Task 27.3: JNI Binding + Scala VideoLoader

**Estimate:** 2h
**Depends on:** 27.2

- `VideoLoader.scala` in `menger-geometry` (or optix-jni if kept generic)
- Native methods: `openVideo(path)`, `videoWidth()`, `videoHeight()`, `frameCount()`,
  `getFrameAt(timestamp): Array[Float]`, `closeVideo()`
- `OptiXVideoApi` trait following existing pattern

---

### Task 27.4: Per-Frame Texture Swap in Animation Loop

**Estimate:** 3h
**Depends on:** 27.3

Extend `WithAnimation.scala` to swap GPU env map texture each frame for video backgrounds.

Frame selection:
```scala
val videoTimestamp = t * scene.envMapVideo.get.totalDuration
val frameData = videoLoader.getFrameAt(videoTimestamp)
renderer.uploadTexture("__video_bg__", frameData, videoLoader.width, videoLoader.height)
renderer.setEnvironmentMap(textureIndex)
```

Streaming per-frame upload (ring buffer preload is a future optimisation).

---

### Task 27.5: DSL `EnvMapVideo` Type

**Estimate:** 1h
**Depends on:** 27.3

```scala
case class EnvMapVideo(
  path: String,
  fps: Double = 25.0,
  loop: Boolean = true,
  startOffset: Double = 0.0,
)
// Scene.envMapVideo: Option[EnvMapVideo] — mutually exclusive with envMap
```

`SceneConfigurator` emits error if both `envMap` and `envMapVideo` are set.

---

### Task 27.6: Documentation

**Estimate:** 2h

- User guide: "Video Backgrounds" section — supported formats, fps vs render fps,
  performance recommendations, HDR video note
- Example scene: 4D sponge in front of time-lapse cloud video
- CHANGELOG.md entry

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 27.1 | ffmpeg CMake dep + VideoLoader skeleton | 3h |
| 27.2 | VideoLoader C++ frame decoder | 4h |
| 27.3 | JNI binding + Scala VideoLoader | 2h |
| 27.4 | Per-frame texture swap in animation loop | 3h |
| 27.5 | DSL EnvMapVideo type | 1h |
| 27.6 | Documentation | 2h |
| **Total** | | **~15h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] `sbt "scalafix --check"` passes
- [ ] CHANGELOG.md updated
- [ ] Test video committed as small `.mp4` for integration tests

---

## Notes

### ffmpeg Licensing

libav (ffmpeg) is LGPL 2.1+. Link as shared library only. Do NOT statically link.

### SDR Only

Sprint 27 supports SDR video (8-bit). HDR10/HLG is future work.
