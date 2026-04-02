# Sprint 17: Animation Tooling, DSL & Architecture Foundations

**Sprint:** 17 - Animation Tooling, DSL & Architecture Foundations
**Status:** Not Started
**Estimate:** ~32 hours
**Branch:** `feature/sprint-17`
**Dependencies:** Sprint 12 (t-Parameter Animation), Sprint 14 (video/preview deferred here)

---

## Goal

Remove the LibGDX rendering path (OptiX-only going forward), introduce a scene graph with
material inheritance, refactor engines to trait composition, add video output, animation
preview, and DSL convenience helpers. Define the optix-jni API boundary for future
library separation.

## Success Criteria

- [ ] LibGDX rendering path fully removed; OptiX is the only renderer
- [ ] Engine uses trait composition (base + `WithAnimation`, `WithPreview`, `WithVideoExport`)
- [ ] Scene graph supports transform hierarchy and per-node material inheritance
- [ ] Video output via ffmpeg: MP4 and WebM from frame sequences
- [ ] CLI: `--video output.mp4` to render and encode in one step
- [ ] Animation preview mode with interactive t scrubbing
- [ ] DSL supports ALL current render settings (width, height, saveName, headless, antialiasing,
  ray depth, camera FOV, caustics parameters, etc.)
- [ ] Procedural placement helpers available in DSL (grids, rings, spirals)
- [ ] Bezier/spline camera path utility implemented as pure Scala helper
- [ ] Runtime DSL scene evaluation works (not just compile-time)
- [ ] optix-jni API boundary documented (general vs. Menger-specific)
- [ ] All tests pass

---

## Tasks

### Task 17.1: Remove LibGDX Rendering Path

**Estimate:** 5h

Remove the dual rendering pipeline. OptiX becomes the sole renderer.

#### What to Remove

- `menger.gdx` package (`GdxRuntime`, `KeyPressTracker`, `DragTracker`, `OrbitCamera`)
- `GdxKeyHandler`, `GdxCameraHandler`
- `MengerEngine`, `InteractiveMengerEngine`
- LibGDX dependency from `build.sbt`
- `--optix` CLI flag (OptiX is now the only mode; flag becomes a no-op or is removed)
- LibGDX-specific code paths in `Main.scala`
- CLI validation rules that reference LibGDX

#### What to Keep

- `OptiXEngine`, `AnimatedOptiXEngine` (refactored in 17.2)
- `OptiXKeyHandler`, `OptiXCameraHandler`
- Camera orbit logic (extract from `OrbitCamera` if needed by OptiX path)

#### Architecture Decision

Update AD-2 (Dual Rendering Pipeline) to "Superseded" — OptiX-only from v0.6.

---

### Task 17.2: Engine Refactor to Trait Composition

**Estimate:** 4h
**Depends on:** 17.1

Replace `OptiXEngine` / `AnimatedOptiXEngine` class hierarchy with a base engine
and composable mix-in traits.

#### Target Design

```scala
// Base engine with render loop, camera, scene setup
trait RenderEngine { ... }

// Mix-in traits for capabilities
trait WithAnimation extends RenderEngine { ... }  // t-parameter sweep, frame export
trait WithPreview extends RenderEngine { ... }    // interactive t-scrubbing
trait WithVideoExport extends RenderEngine { ... } // ffmpeg pipeline

// Composed engines
class InteractiveEngine extends RenderEngine
class AnimationEngine extends RenderEngine with WithAnimation
class PreviewEngine extends RenderEngine with WithPreview
class VideoEngine extends RenderEngine with WithAnimation with WithVideoExport
```

#### Files to Modify

- `menger-app/src/main/scala/menger/engines/OptiXEngine.scala` — extract base trait
- `menger-app/src/main/scala/menger/engines/AnimatedOptiXEngine.scala` — convert to trait
- `menger-app/src/main/scala/Main.scala` — compose engines from traits

#### Files to Create

- Trait files for `WithAnimation`, `WithPreview`, `WithVideoExport`

---

### Task 17.3: Scene Graph

**Estimate:** 5h

Replace the flat `List[SceneObject]` with a tree of scene nodes supporting transform
hierarchy and per-node material inheritance.

#### Data Model

```scala
case class SceneNode(
  transform: Transform = Transform.identity,  // local position, rotation, scale
  material: Option[Material] = None,          // overrides parent if present
  geometry: Option[Geometry] = None,          // leaf node geometry
  children: List[SceneNode] = Nil
)
```

Children inherit the nearest ancestor's material unless they specify their own.
Transforms accumulate down the tree (child's world transform = parent's world * local).

#### Files to Create

- `menger-app/src/main/scala/menger/dsl/SceneNode.scala`
- `menger-app/src/main/scala/menger/dsl/Transform.scala`

#### Files to Modify

- `menger-app/src/main/scala/menger/dsl/Scene.scala` — root node instead of object list
- `menger-app/src/main/scala/menger/dsl/SceneConverter.scala` — tree traversal, transform
  accumulation, material resolution
- DSL syntax for nesting/grouping

---

### Task 17.4: All Render Settings in DSL

**Estimate:** 4h

Make ALL current CLI render settings expressible in the Scala DSL. Precedence:
CLI overrides DSL (CLI is the "I know what I want right now" path).

#### Settings to Add

- Window: `width`, `height`, `saveName`, `headless`
- Rendering: `antialiasing` (samples), `maxRayDepth`, `backgroundColor`
- Camera: `fov`, `position`, `lookAt`, `up`
- Caustics: `photonsPerIteration`, `iterations`, `initialRadius`, `alpha`
- Materials: default material settings
- Any other current CLI-only settings

#### Precedence Model

```
Final setting = CLI value if provided, else DSL value if provided, else default
```

---

### Task 17.5: Video Output via ffmpeg

**Estimate:** 4h

Wrap frame sequences produced by the animation engine into video files using ffmpeg.

#### Proposed CLI

```bash
menger --scene examples.dsl.OrbitingSphere \
  --frames 120 --start-t 0 --end-t 6.28 \
  --video orbit.mp4

menger --scene examples.dsl.OrbitingSphere \
  --frames 120 --start-t 0 --end-t 6.28 \
  --video orbit.webm --video-quality 28
```

#### Implementation

- `VideoEncoder` class wrapping ffmpeg via `ProcessBuilder`
- Detect ffmpeg availability at startup
- Support MP4 (H.264) and WebM (VP9) based on file extension
- Configurable quality (CRF value)
- Clean up temporary frame files after encoding (optional `--keep-frames`)

---

### Task 17.6: Animation Preview Mode

**Estimate:** 3h
**Depends on:** 17.2

Interactive mode where the user can scrub through t values in real-time.

#### Proposed UX

```bash
menger --scene examples.dsl.OrbitingSphere --preview
```

- Left/Right arrow keys: step t by 0.01
- Shift+Left/Right: step t by 0.1
- Home/End: jump to start-t / end-t
- Space: play/pause automatic t sweep
- Mouse scroll: fine-tune t value
- On-screen display: current t value, frame number

Implemented as the `WithPreview` trait from 17.2.

---

### Task 17.7: Runtime DSL Scene Evaluation

**Estimate:** 2h

Allow `.scala` scene files to be compiled and evaluated at runtime, enabling hot-reload
and interactive iteration without recompiling the application.

#### Mechanism

- Add `scala-compiler` dependency
- `SceneLoader` gains ability to compile a `.scala` file at runtime
- Extract the `Scene` object from the compiled class
- Error reporting for compilation failures

---

### Task 17.8: Bezier/Spline Camera Path Utility

**Estimate:** 2h

Pure Scala helper for building smooth camera paths as a function of `t`.
Useful with the existing `scene(t)` animation system.

---

### Task 17.9: Procedural Placement Helpers in DSL

**Estimate:** 2h
**Depends on:** 17.3 (scene graph)

Helpers for procedural object placement using the scene graph:
- Grids, rings, spirals, random distributions
- Each placement creates a subtree of `SceneNode`s with appropriate transforms

---

### Task 17.10: Define optix-jni API Boundary

**Estimate:** 1h

Design-only task. Document which current optix-jni methods are general-purpose OptiX
operations vs. Menger-specific convenience methods.

#### Deliverable

A document (in `optix-jni/docs/` or `docs/arc42/`) classifying current API surface:
- **General:** `initialize()`, `dispose()`, `render()`, `setCamera()`, `setLight()`
- **Menger-specific:** `setSphere()`, `setSphereColor()`, `setSphereIOR()`
- **To generalize:** `setMesh()` (general concept, Menger-specific parameters)

New features from Sprint 17+ should be added on the correct side of this boundary.

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 17.1 | Remove LibGDX rendering path | 5h | None |
| 17.2 | Engine refactor to trait composition | 4h | 17.1 |
| 17.3 | Scene graph (transforms + material inheritance) | 5h | None |
| 17.4 | All render settings in DSL | 4h | None |
| 17.5 | Video output via ffmpeg | 4h | 17.2 |
| 17.6 | Animation preview / t-scrubbing | 3h | 17.2 |
| 17.7 | Runtime DSL scene evaluation | 2h | None |
| 17.8 | Bezier/spline camera path utility | 2h | None |
| 17.9 | Procedural placement helpers | 2h | 17.3 |
| 17.10 | Define optix-jni API boundary | 1h | None |
| **Total** | | **~32h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] arc42 updated: AD-2 superseded, new AD for scene graph and engine traits
- [ ] `docs/guide/advanced.md` and `docs/guide/dsl-reference.md` updated

---

## Notes

### Background

Sprint 17 combines animation tooling (deferred from Sprint 14) with architectural
foundations that simplify all future work:

- **LibGDX removal** eliminates the dual-pipeline complexity that pervades the engine,
  input handling, and camera systems.
- **Engine traits** prevent the class hierarchy from growing with each new rendering mode.
- **Scene graph** enables composition, grouping, and material inheritance needed by
  Sprint 19+ (advanced geometry, procedural placement).
- **optix-jni boundary** ensures future features land on the correct side of the
  library separation.

### Removed from Original Plan

- **17.8 (Animation JSON export/import)** — dropped. Runtime Scala DSL evaluation (17.7)
  provides a more expressive alternative. If crash recovery for partial animations is
  needed, a simple checkpoint file (last completed frame number) suffices.

**MILESTONE: v0.6 — Animation & Architecture Foundations**
