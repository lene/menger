# Engine Trait Composition Design (Task 17.2)

**Date:** 2026-04-02  
**Sprint:** 17  
**Status:** Approved

---

## Problem

`OptiXEngine` and `AnimatedOptiXEngine` are independent classes that duplicate scene-building
logic (classifying scenes, dispatching to sphere vs mesh builders, clearing instances). Adding
each new rendering mode (video, preview) requires a new class that must replicate this
infrastructure. `RenderEngine` is coupled to LibGDX's `Game` class, making it impossible to
test or use without a LibGDX runtime.

---

## Goal

Replace `OptiXEngine` / `AnimatedOptiXEngine` with a composable trait hierarchy that:

1. Makes `RenderEngine` a pure Scala lifecycle interface (no LibGDX dependency)
2. Houses shared rendering infrastructure in one `BaseEngine` abstract class
3. Expresses additional capabilities (animation, preview, video) as mix-in traits
4. Eliminates scene-building duplication between the two existing engines
5. Establishes stub traits for capabilities to be implemented in tasks 17.5 and 17.6

---

## Non-Goals

- No change to rendering behaviour (this is a structural refactor)
- No new CLI flags
- `WithPreview` and `WithVideoExport` stubs have no behaviour yet
- No new performance baselines required

---

## Architecture

### Layer Overview

```
trait RenderEngine          -- pure Scala lifecycle interface (no LibGDX)
  |
abstract class BaseEngine   -- extends Game with RenderEngine
  |                            shared: rendererWrapper, renderResources, cameraState,
  |                            scene-building primitives
  |
  +-- class InteractiveEngine           -- was OptiXEngine
  +-- class AnimationEngine             -- was AnimatedOptiXEngine
  |     with WithAnimation
  +-- class PreviewEngine  (stub)
  |     with WithPreview
  +-- class VideoEngine    (stub)
        with WithAnimation
        with WithVideoExport
```

### Capability Traits

```
trait WithAnimation   -- t-parameter sweep, frame counter, calls BaseEngine render primitives
trait WithPreview     -- stub (Task 17.6 fills in interactive t-scrubbing)
trait WithVideoExport -- stub (Task 17.5 fills in ffmpeg pipeline)
```

`WithAnimation` uses a self-type `this: BaseEngine =>` to access `rendererWrapper`,
`renderResources`, and `cameraState` without duplicating them.

---

## Component Specifications

### `RenderEngine` (modified)

**File:** `menger-app/src/main/scala/menger/engines/RenderEngine.scala`

Remove `extends Game`. Becomes a pure Scala trait declaring the lifecycle interface:

```scala
trait RenderEngine:
  def create(): Unit
  def render(): Unit
  def resize(width: Int, height: Int): Unit
  def dispose(): Unit
  def pause(): Unit
  def resume(): Unit
```

No imports from LibGDX. Independently testable.

---

### `BaseEngine` (new)

**File:** `menger-app/src/main/scala/menger/engines/BaseEngine.scala`

Abstract class that bridges `Game` (LibGDX lifecycle) and `RenderEngine` (our interface).
Holds all infrastructure shared by concrete engines:

```scala
abstract class BaseEngine(maxInstances: Int)(using ProfilingConfig)
    extends Game with RenderEngine with LazyLogging:

  protected val rendererWrapper: OptiXRendererWrapper
  protected val renderResources: OptiXRenderResources
  protected val sceneConfigurator: SceneConfigurator
  protected val cameraState: CameraState

  // Shared scene-building
  protected def buildSceneFromSpecs(specs: List[ObjectSpec], renderer: OptiXRenderer): Try[Unit]
  protected def buildSceneFromConfigs(configs: SceneConfigs, renderer: OptiXRenderer): Try[Unit]
  protected def rebuildGeometry(specs: List[ObjectSpec], renderer: OptiXRenderer): Unit

  // Default lifecycle stubs (concrete engines override what they need)
  override def resize(width: Int, height: Int): Unit = {}
  override def dispose(): Unit =
    renderResources.dispose()
    rendererWrapper.dispose()
  override def pause(): Unit = {}
  override def resume(): Unit = {}
```

The `OptiXEngineConfig` is not a `BaseEngine` concern — each concrete engine's constructor
takes exactly the parameters it needs.

**Scene-building logic moved here from both old engines:**
- `buildSceneFromSpecs`: classifies scene type, dispatches to sphere/mesh builders
  (extracted from `OptiXEngine.buildInitialGeometry`)
- `buildSceneFromConfigs`: same but from `SceneConverter.SceneConfigs`
  (extracted from `AnimatedOptiXEngine.buildSceneFromConfigs`)
- `rebuildGeometry`: clears instances, rebuilds
  (extracted from `OptiXEngine.rebuildGeometry`)

---

### `WithAnimation` (new)

**File:** `menger-app/src/main/scala/menger/engines/WithAnimation.scala`

Mix-in trait providing t-parameter animation logic. Uses self-type `this: BaseEngine =>`.

```scala
trait WithAnimation extends RenderEngine:
  self: BaseEngine =>

  protected def sceneFunction: Float => Scene
  protected def animConfig: TAnimationConfig

  // Concrete: frameCounter (AtomicInteger), per-frame render+save loop
  // Hooks into self.rendererWrapper, self.renderResources, self.cameraState
```

The `render()` method provided by this trait loops through frames, evaluating
`sceneFunction(t)` per frame, rebuilding geometry via `self.buildSceneFromConfigs`,
rendering via `self.rendererWrapper`, and calling `saveImage()`. Exits after last frame.

---

### `WithPreview` (new, stub)

**File:** `menger-app/src/main/scala/menger/engines/WithPreview.scala`

```scala
trait WithPreview extends RenderEngine:
  // Stub: filled in by Task 17.6
  // Will add: t-scrubbing key bindings, on-screen t display
  // Left/Right keys step t, Space toggles play/pause
```

---

### `WithVideoExport` (new, stub)

**File:** `menger-app/src/main/scala/menger/engines/WithVideoExport.scala`

```scala
trait WithVideoExport extends RenderEngine:
  // Stub: filled in by Task 17.5
  // Will add: ffmpeg pipeline invoked after animation completes
```

---

### `InteractiveEngine` (renamed from `OptiXEngine`)

**File:** `menger-app/src/main/scala/menger/engines/InteractiveEngine.scala`  
**Old file deleted:** `OptiXEngine.scala`

```scala
class InteractiveEngine(config: OptiXEngineConfig, userSetMaxInstances: Boolean = false)
    (using ProfilingConfig)
    extends BaseEngine(config.execution.maxInstances)
    with TimeoutSupport with SavesScreenshots with Observer:
  ...
```

All existing functionality of `OptiXEngine` is preserved: 4D rotation event handling,
keyboard input, camera control, interactive re-render on change, timeout support,
screenshot saving, and non-interactive (single-render + exit) mode.

The scene-building methods are now calls to `BaseEngine` primitives rather than
self-contained implementations.

---

### `AnimationEngine` (renamed from `AnimatedOptiXEngine`)

**File:** `menger-app/src/main/scala/menger/engines/AnimationEngine.scala`  
**Old file deleted:** `AnimatedOptiXEngine.scala`

```scala
class AnimationEngine(
    sceneFunction: Float => Scene,
    animConfig: TAnimationConfig,
    executionConfig: ExecutionConfig,
    renderConfig: RenderConfig,
    causticsConfig: CausticsConfig
)(using ProfilingConfig)
    extends BaseEngine(executionConfig.maxInstances)
    with WithAnimation with SavesScreenshots:
  ...
```

The `render()` loop and frame management logic move to `WithAnimation`. `AnimationEngine`
provides the constructor parameters and delegates to the trait.

---

### `PreviewEngine` (new, stub)

**File:** `menger-app/src/main/scala/menger/engines/PreviewEngine.scala`

```scala
class PreviewEngine(/* TBD by Task 17.6 */)(using ProfilingConfig)
    extends BaseEngine(???) with WithPreview:
  override def create(): Unit = ???
  override def render(): Unit = ???
```

Constructor signature is a placeholder; Task 17.6 fills in the implementation.

---

### `VideoEngine` (new, stub)

**File:** `menger-app/src/main/scala/menger/engines/VideoEngine.scala`

```scala
class VideoEngine(
    sceneFunction: Float => Scene,
    animConfig: TAnimationConfig,
    executionConfig: ExecutionConfig,
    renderConfig: RenderConfig,
    causticsConfig: CausticsConfig,
    videoOutputPath: String
)(using ProfilingConfig)
    extends BaseEngine(executionConfig.maxInstances)
    with WithAnimation with WithVideoExport:
  ...
```

Constructor mirrors `AnimationEngine` plus `videoOutputPath`. Animation half works immediately
(via `WithAnimation`); video encoding is a no-op stub until Task 17.5.

---

### `Main.scala` (modified)

Update engine construction to use renamed classes:
- `OptiXEngine(...)` → `InteractiveEngine(...)`
- `AnimatedOptiXEngine(...)` → `AnimationEngine(...)`

No logic changes.

---

## Data Flow

```
MengerCLIOptions
  → Main.createEngine()
      → InteractiveEngine(config)    # --objects, --scene (static)
      → AnimationEngine(fn, config)  # --scene (animated) + --frames
      → PreviewEngine (stub)         # --preview (not yet wired)
      → VideoEngine (stub)           # --video (not yet wired)
  → Lwjgl3Application(engine, lwjglConfig)
      → engine.create()
      → engine.render() (loop)
      → engine.dispose()
```

---

## Testing

### New: `RenderEngineSuite`

**File:** `menger-app/src/test/scala/menger/engines/RenderEngineSuite.scala`

Verifies `RenderEngine` is a standalone trait with no LibGDX import. Creates a test double
(minimal concrete implementation) and exercises all lifecycle method signatures.

```scala
class TestEngine extends RenderEngine:
  override def create(): Unit = ()
  override def render(): Unit = ()
  override def resize(w: Int, h: Int): Unit = ()
  override def dispose(): Unit = ()
  override def pause(): Unit = ()
  override def resume(): Unit = ()

"RenderEngine" should "be implementable without LibGDX" in:
  val engine = new TestEngine
  engine.create()
  engine.render()
  engine.dispose()
```

### New: `WithAnimationSuite`

**File:** `menger-app/src/test/scala/menger/engines/WithAnimationSuite.scala`

Tests the pure-logic portions of `WithAnimation` that don't require a GPU:
- `TAnimationConfig.tForFrame()` arithmetic (complementing the existing `TAnimationConfigSuite`)
- Frame-completion predicate: `frame >= animConfig.frames`
- `currentSaveName` pattern formatting: `String.format(savePattern, frameIndex)` correctness

Note: `WithAnimation.render()` calls `self.rendererWrapper` (GPU), so full render-loop
tests remain integration-level. This suite covers only the config and frame-logic
computations that can be expressed as pure functions on `TAnimationConfig`.

### Updated: `InteractiveEngineSuite` (was `OptiXEngineSuite`)

**File:** `menger-app/src/test/scala/menger/InteractiveEngineSuite.scala`  
**Old file:** `menger-app/src/test/scala/menger/OptiXEngineSuite.scala`

Rename class references from `OptiXEngine` → `InteractiveEngine`. All existing test
expectations remain valid — this is purely a structural rename.

---

## Files Summary

### New files

| File | Type | Description |
|------|------|-------------|
| `engines/BaseEngine.scala` | Abstract class | Shared infrastructure, scene building |
| `engines/WithAnimation.scala` | Trait | t-sweep animation loop |
| `engines/WithPreview.scala` | Trait (stub) | Preview mode interface |
| `engines/WithVideoExport.scala` | Trait (stub) | Video export interface |
| `engines/InteractiveEngine.scala` | Class | Replaces OptiXEngine |
| `engines/AnimationEngine.scala` | Class | Replaces AnimatedOptiXEngine |
| `engines/PreviewEngine.scala` | Class (stub) | Wired in Task 17.6 |
| `engines/VideoEngine.scala` | Class (stub) | Wired in Task 17.5 |
| `test/.../RenderEngineSuite.scala` | Test | RenderEngine interface test |
| `test/.../WithAnimationSuite.scala` | Test | WithAnimation logic test |

### Modified files

| File | Change |
|------|--------|
| `engines/RenderEngine.scala` | Remove `extends Game` |
| `Main.scala` | Use `InteractiveEngine`, `AnimationEngine` |
| `test/.../InteractiveEngineSuite.scala` | Rename from OptiXEngineSuite, update refs |

### Deleted files

| File | Replaced by |
|------|-------------|
| `engines/OptiXEngine.scala` | `InteractiveEngine.scala` + `BaseEngine.scala` |
| `engines/AnimatedOptiXEngine.scala` | `AnimationEngine.scala` + `WithAnimation.scala` |
| `test/.../OptiXEngineSuite.scala` | `InteractiveEngineSuite.scala` |

---

## Constraints

- Scala 3 only (no Scala 2 syntax)
- No `var` (Wartremover); use `AtomicReference` or `AtomicInteger` for mutable counters
- No `null`, no `throw` in production code
- Max 100 chars per line
- Alphabetical import ordering (Scalafix)
- All tests must pass before commit
