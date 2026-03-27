# Sprint 17: Animation Tooling & DSL

**Sprint:** 17 - Animation Tooling & DSL
**Status:** Not Started
**Estimate:** ~19 hours
**Branch:** `feature/sprint-17`
**Dependencies:** Sprint 12 (t-Parameter Animation), Sprint 14 (video/preview deferred here)

---

## Goal

Add video output, animation preview, and DSL convenience helpers for animation and scene
composition. Plain Scala in `scene(t)` already serves as the animation DSL — this sprint
adds convenience helpers, tooling, and the video/preview pipeline deferred from Sprint 14.

## Success Criteria

- [ ] Video output via ffmpeg: MP4 and WebM from frame sequences
- [ ] CLI: `--video output.mp4` to render and encode in one step
- [ ] Animation preview mode with interactive t scrubbing
- [ ] DSL supports setting window width, height, saveName, and headless mode
- [ ] Scene composition helpers available in DSL (SceneBuilder utilities)
- [ ] Procedural placement helpers available in DSL
- [ ] Bezier/spline camera path utility implemented as pure Scala helper
- [ ] Runtime DSL scene evaluation works (not just compile-time)
- [ ] Animation export/import (JSON format for t-param frame configs)
- [ ] All tests pass

---

## Tasks

### Task 17.1: Video Output via ffmpeg

**Estimate:** 4h

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

- `menger-app/src/main/scala/menger/MengerCLIOptions.scala` — add `--video`, `--video-quality`, `--video-fps`, `--keep-frames`
- `menger-app/src/main/scala/menger/cli/CliValidation.scala` — validate video options
- `menger-app/src/main/scala/menger/engines/AnimatedOptiXEngine.scala` — invoke VideoEncoder after all frames rendered

---

### Task 17.2: Animation Preview Mode

**Estimate:** 3h

Interactive mode where the user can scrub through t values in real-time using keyboard/mouse.

#### Proposed UX

```bash
menger --optix --scene examples.dsl.OrbitingSphere --preview
```

- Left/Right arrow keys: step t by 0.01
- Shift+Left/Right: step t by 0.1
- Home/End: jump to start-t / end-t
- Space: play/pause automatic t sweep
- Mouse scroll: fine-tune t value
- On-screen display: current t value, frame number

#### Files to Create

- `menger-app/src/main/scala/menger/engines/PreviewOptiXEngine.scala`

#### Files to Modify

- `menger-app/src/main/scala/menger/input/OptiXKeyHandler.scala` — add preview key bindings
- `menger-app/src/main/scala/Main.scala` — wire --preview option
- `menger-app/src/main/scala/menger/MengerCLIOptions.scala` — add `--preview`

---

### Task 17.3: DSL Window/Output Settings

**Estimate:** 2h

Add `width`, `height`, `saveName`, and `headless` to the DSL configuration API
(currently CLI-only).

---

### Task 17.4: Scene Composition Helpers

**Estimate:** 2h

Add `SceneBuilder` utilities for common scene construction patterns (grouping, instancing,
positioning).

---

### Task 17.5: Procedural Placement Helpers in DSL

**Estimate:** 2h

Add helpers for procedural object placement (grids, rings, spirals, random distributions).

---

### Task 17.6: Bezier/Spline Camera Path Utility

**Estimate:** 2h

Pure Scala helper (no engine changes) for building smooth camera paths as a function of `t`.
Useful with the existing `scene(t)` animation system.

---

### Task 17.7: Runtime DSL Scene Evaluation

**Estimate:** 2h

Allow scenes to be evaluated at runtime (not just compiled in). Enables hot-reload and
interactive iteration without recompiling.

---

### Task 17.8: Animation Export/Import (JSON)

**Estimate:** 2h

JSON format for saving/loading t-parameter frame configurations (frame count, range, output
paths). Enables repeatable animation runs from config files.

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 17.1 | Video output via ffmpeg | 4h |
| 17.2 | Animation preview / t-scrubbing | 3h |
| 17.3 | DSL window/output settings | 2h |
| 17.4 | Scene composition helpers | 2h |
| 17.5 | Procedural placement helpers | 2h |
| 17.6 | Bezier/spline camera path utility | 2h |
| 17.7 | Runtime DSL evaluation | 2h |
| 17.8 | Animation export/import (JSON) | 2h |
| **Total** | | **~19h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] `docs/guide/advanced.md` and `docs/guide/dsl-reference.md` updated (video output guide, animation preview, new DSL helpers)

---

## Notes

### Background

Tasks 17.1 and 17.2 were deferred from Sprint 14 to keep that sprint focused on rendering
correctness. Tasks 17.3–17.8 complete the DSL convenience work originally planned for
Sprint 15, merging deferred Sprint 10 DSL work with the animation pipeline additions.

The t-parameter animation system (Sprint 12) eliminated the need for a separate
animation DSL — `scene(t)` is the DSL. This sprint adds ergonomic helpers on top.
