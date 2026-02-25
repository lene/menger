# Sprint 14: Video Output & Visual Enhancements

**Sprint:** 14 - Video Output & Visual Enhancements
**Status:** Not Started
**Estimate:** 15-20 hours
**Branch:** `feature/sprint-14`
**Dependencies:** Sprint 12 (t-Parameter Animation) - required, Sprint 13 (Visual Quality) - optional

> **Note:** This sprint replaces the original Sprint 14 ("Advanced Animation System") which was built on the keyframe-based animation approach from the original Sprint 13. With the t-parameter animation system (Sprint 12), most of the old Sprint 14 content is obsolete:
>
> **Obsolete** (handled naturally by `scene(t)`):
> - Easing functions (users implement their own easing in `scene(t)`)
> - Per-instance color/IOR animation (rebuilt each frame from `scene(t)`)
> - Generic AnimatableProperty system (unnecessary with functional approach)
> - Extended DSL animation syntax (users write plain Scala)
> - Camera/light animation (part of the Scene returned by `scene(t)`)
>
> **Retained and expanded below:**
> - Video output via ffmpeg (wrapping frame sequences into MP4/WebM)
> - Animation preview mode (scrubbing through t values interactively)
> - Visual enhancements from old Sprint 15 that fit naturally here

---

## Goal

Add video output from t-parameter animation frame sequences, an interactive preview mode for scrubbing through t values, and visual enhancements including soft shadows, depth of field, and additional geometric primitives.

## Success Criteria

- [ ] Video output via ffmpeg: MP4 and WebM from frame sequences
- [ ] CLI: `--video output.mp4` to render and encode in one step
- [ ] Animation preview mode with interactive t scrubbing
- [ ] Soft shadows with area lights (penumbra)
- [ ] Depth of field (camera aperture, bokeh)
- [ ] Additional primitives: cylinder, cone, torus
- [ ] Coordinate cross (axis visualization for debugging)
- [ ] All tests pass (~25-30 new tests)

---

## Tasks

### Task 14.1: Video Output via ffmpeg

**Estimate:** 4 hours

Wrap frame sequences produced by `AnimatedOptiXEngine` into video files using ffmpeg.

#### Proposed CLI

```bash
# Render animation directly to MP4
menger --optix --scene examples.dsl.OrbitingSphere \
  --frames 120 --start-t 0 --end-t 6.28 \
  --video orbit.mp4

# Render animation directly to WebM
menger --optix --scene examples.dsl.OrbitingSphere \
  --frames 120 --start-t 0 --end-t 6.28 \
  --video orbit.webm --video-quality 28

# Render frames first, then encode separately
menger --optix --scene examples.dsl.OrbitingSphere \
  --frames 120 --start-t 0 --end-t 6.28 \
  --save-name /tmp/frames/orbit_%04d.png
# Then manually:
ffmpeg -framerate 30 -i /tmp/frames/orbit_%04d.png -c:v libx264 -pix_fmt yuv420p orbit.mp4
```

#### Implementation

- `VideoEncoder` class wrapping ffmpeg via `ProcessBuilder`
- Detect ffmpeg availability at startup
- Support MP4 (H.264) and WebM (VP9) based on file extension
- Configurable quality (CRF value)
- Clean up temporary frame files after encoding (optional `--keep-frames`)
- Progress reporting during encoding

#### Files to Create

- `menger-app/src/main/scala/menger/engines/VideoEncoder.scala`

#### Files to Modify

- `menger-app/src/main/scala/menger/MengerCLIOptions.scala` -- add `--video`, `--video-quality`, `--video-fps`, `--keep-frames`
- `menger-app/src/main/scala/menger/cli/CliValidation.scala` -- validate video options
- `menger-app/src/main/scala/menger/engines/AnimatedOptiXEngine.scala` -- invoke VideoEncoder after all frames rendered

#### Tests

- VideoEncoder config validation (format detection, quality range)
- CLI option validation (--video requires --frames, --video requires --scene)
- ffmpeg availability detection (graceful failure if not installed)

---

### Task 14.2: Animation Preview Mode

**Estimate:** 3 hours

Interactive mode where the user can scrub through t values in real-time using keyboard/mouse.

#### Proposed UX

```bash
# Open interactive preview for an animated scene
menger --optix --scene examples.dsl.OrbitingSphere --preview
```

- Left/Right arrow keys: step t by 0.01
- Shift+Left/Right: step t by 0.1
- Home/End: jump to start-t / end-t
- Space: play/pause automatic t sweep
- Mouse scroll: fine-tune t value
- On-screen display: current t value, frame number

#### Implementation

- Extend `OptiXKeyHandler` with preview controls
- On-screen HUD showing current t value
- Automatic playback mode (step t per frame at configurable speed)
- Full scene rebuild on each t change (same as AnimatedOptiXEngine)

#### Files to Create

- `menger-app/src/main/scala/menger/engines/PreviewOptiXEngine.scala`

#### Files to Modify

- `menger-app/src/main/scala/menger/input/OptiXKeyHandler.scala` -- add preview key bindings
- `menger-app/src/main/scala/Main.scala` -- wire --preview option
- `menger-app/src/main/scala/menger/MengerCLIOptions.scala` -- add `--preview`

---

### Task 14.3: Soft Shadows with Area Lights

**Estimate:** 3 hours

Replace point/directional lights with area lights that produce penumbra (soft shadow edges).

#### Implementation

- Add `AreaLight` type to DSL: `AreaLight(position, size, intensity, color)`
- Multiple shadow rays per pixel sampling the light area
- Configurable shadow samples (default: 4, max: 16)
- CLI: `--shadow-samples N`

#### Files to Modify

- `menger-app/src/main/scala/menger/dsl/Light.scala` -- add `AreaLight` case
- `optix-jni/src/main/native/shaders/shadows.cu` -- multi-sample shadow rays
- `optix-jni/src/main/native/include/OptiXData.h` -- area light data structure

---

### Task 14.4: Depth of Field

**Estimate:** 3 hours

Camera aperture simulation producing bokeh (out-of-focus blur).

#### CLI

```bash
menger --optix --aperture 0.1 --focus-distance 5.0
```

#### Implementation

- Jitter camera ray origins within aperture disk
- Focus plane at specified distance
- Multiple samples per pixel (reuse AA infrastructure)
- Configurable aperture size and focus distance

#### Files to Modify

- `optix-jni/src/main/native/shaders/raygen_primary.cu` -- add aperture jitter
- `menger-app/src/main/scala/menger/MengerCLIOptions.scala` -- add `--aperture`, `--focus-distance`

---

### Task 14.5: Additional Primitives

**Estimate:** 3 hours

Add cylinder, cone, and torus geometry types for richer scene composition.

#### DSL Syntax

```scala
Scene(
  objects = List(
    Cylinder(pos = (0f, 0f, 0f), radius = 0.5f, height = 2f, material = Material.Chrome),
    Cone(pos = (2f, 0f, 0f), radius = 0.5f, height = 1.5f, material = Material.Gold),
    Torus(pos = (-2f, 0f, 0f), majorRadius = 1f, minorRadius = 0.3f, material = Material.Glass)
  ),
  // ...
)
```

#### Implementation

- Triangle mesh generation for each primitive
- Parametric UV coordinates for texturing
- Normal computation for smooth shading
- Integration with existing scene builder system

#### Files to Create

- `menger-app/src/main/scala/menger/dsl/Cylinder.scala` (or extend SceneObject)
- `menger-app/src/main/scala/menger/dsl/Cone.scala`
- `menger-app/src/main/scala/menger/dsl/Torus.scala`

---

### Task 14.6: Coordinate Cross (Axis Visualization)

**Estimate:** 1.5 hours

Render XYZ axis lines for debugging scene layout and camera positioning.

#### CLI

```bash
menger --optix --show-axes          # Show coordinate cross at origin
menger --optix --show-axes --axis-length 5.0  # Custom axis length
```

#### Implementation

- Render thin cylinders along X (red), Y (green), Z (blue) axes
- Configurable length and thickness
- Toggle on/off via CLI or keyboard shortcut

---

### Task 14.7: Documentation and Examples

**Estimate:** 1.5 hours

Update documentation for all new features.

#### Sections to Add

- Video output guide (ffmpeg setup, format options)
- Animation preview mode usage
- Soft shadows examples
- Depth of field photography guide
- New primitive reference

---

## Summary

| Task | Description | Estimate | Priority |
|------|-------------|----------|----------|
| 14.1 | Video output via ffmpeg | 4h | High |
| 14.2 | Animation preview mode | 3h | High |
| 14.3 | Soft shadows (area lights) | 3h | Medium |
| 14.4 | Depth of field | 3h | Medium |
| 14.5 | Additional primitives | 3h | Medium |
| 14.6 | Coordinate cross | 1.5h | Low |
| 14.7 | Documentation | 1.5h | High |
| **Total** | | **19h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] USER_GUIDE.md updated
- [ ] Example scenes created and tested
- [ ] Integration tests cover new features

---

## Notes

### Implementation Order

Recommended order:

1. **Task 14.1** (Video output) - High value, extends Sprint 12 animation
2. **Task 14.2** (Preview mode) - High value for iterating on animated scenes
3. **Task 14.5** (Primitives) - Enriches scene composition
4. **Task 14.3** (Soft shadows) - Visual quality improvement
5. **Task 14.4** (Depth of field) - Visual quality improvement
6. **Task 14.6** (Coordinate cross) - Debugging aid
7. **Task 14.7** (Documentation) - Last

### Relationship to Previous Plans

The t-parameter animation system (Sprint 12) made the original keyframe-based animation approach (old Sprints 13-14) obsolete. With `scene(t)`, users have full control over every aspect of the scene at every point in time using plain Scala code. This is more expressive than a keyframe system and requires no special animation DSL.

What remains valuable from the old plans:
- **Video output** -- Users still need a convenient way to encode frame sequences
- **Preview mode** -- Interactive exploration of the t-parameter space
- **Visual enhancements** -- Soft shadows, depth of field, and new primitives improve rendering quality regardless of animation approach
