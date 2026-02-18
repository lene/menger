# Plan: Task 11.1 — Scala Wrapper for libGDX

## Goal

Eliminate all `var` and `null` from application-layer Scala code by introducing a wrapper package `menger.gdx` with 4 new classes. After this change, `var` and `null` exist only inside the wrapper layer.

## Current State

**9 `var` declarations** across 5 files (all `@SuppressWarnings`-annotated):

| File | var | Type |
|------|-----|------|
| `OptiXEngine.scala:70` | `currentObjectSpecs` | `Option[List[ObjectSpec]]` |
| `OptiXKeyHandler.scala:27` | `rotatePressed` | `Map[Key, Boolean]` |
| `GdxKeyHandler.scala:32` | `rotatePressed` | `Map[Key, Boolean]` (identical) |
| `GdxCameraHandler.scala:35` | `shiftStart` | `ScreenCoords` |
| `OptiXCameraHandler.scala:48` | `eye` | `Vector3` |
| `OptiXCameraHandler.scala:51` | `lookAt` | `Vector3` |
| `OptiXCameraHandler.scala:54` | `up` | `Vector3` |
| `OptiXCameraHandler.scala:65` | `spherical` | `SphericalCoords` |
| `OptiXCameraHandler.scala:80` | `dragState` | `Option[CameraDragState]` |

**6 `null` checks** (all `scalafix:off/on`-annotated):

| File | Context |
|------|---------|
| `OptiXKeyHandler.scala:36` | `Gdx.app != null` (Escape) |
| `OptiXKeyHandler.scala:41` | `Gdx.app != null` (Ctrl+Q) |
| `OptiXKeyHandler.scala:61` | `Gdx.graphics != null` (update) |
| `GdxKeyHandler.scala:44` | `Gdx.app != null` (Ctrl+Q) |
| `OptiXEngine.scala:185` | `Gdx.app.exit()` (unguarded) |
| `OptiXEngine.scala:442` | `Gdx.app.exit()` (unguarded) |

**Out of scope**: `modifierState` var in `InputHandler.scala:17` — pure domain state, not libGDX-related.

---

## New Files (package `menger.gdx`)

### 1. `GdxRuntime.scala` — null-safe Gdx global access

Consolidates ALL `Gdx.app`/`Gdx.graphics`/`Gdx.input`/`Gdx.gl` null checks into one file.

```scala
package menger.gdx

object GdxRuntime:
  // scalafix:off DisableSyntax.null  (single consolidated pair)
  private def app = Option(Gdx.app)
  private def graphics = Option(Gdx.graphics)
  private def input = Option(Gdx.input)
  private def gl = Option(Gdx.gl)
  // scalafix:on DisableSyntax.null

  def exit(): Unit = app.foreach(_.exit())
  def requestRendering(): Unit = graphics.foreach(_.requestRendering())
  def setContinuousRendering(v: Boolean): Unit = graphics.foreach(_.setContinuousRendering(v))
  def deltaTime: Float = graphics.map(_.getDeltaTime).getOrElse(0f)
  def width: Int = graphics.map(_.getWidth).getOrElse(0)
  def height: Int = graphics.map(_.getHeight).getOrElse(0)
  def glClear(mask: Int): Unit = gl.foreach(_.glClear(mask))
  def setInputProcessor(p: InputProcessor): Unit = input.foreach(_.setInputProcessor(p))
  def isKeyPressed(keycode: Int): Boolean = input.exists(_.isKeyPressed(keycode))
  def isButtonPressed(button: Int): Boolean = input.exists(_.isButtonPressed(button))
```

### 2. `KeyPressTracker.scala` — encapsulates key-press var + deduplicates

Used by both `OptiXKeyHandler` and `GdxKeyHandler` (identical pattern today).

```scala
class KeyPressTracker:
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var pressed: Map[Key, Boolean] = Map().withDefaultValue(false)

  def press(key: Key): Unit = pressed = pressed.updated(key, true)
  def release(key: Key): Unit = pressed = pressed.updated(key, false)
  def isPressed(key: Key): Boolean = pressed(key)
  def anyPressed: Boolean = pressed.values.exists(identity)
```

### 3. `OrbitCamera.scala` — encapsulates 5 camera vars

Mixes in `SphericalOrbit`. Owns all mutable camera state. Provides `orbit()`, `pan()`, `zoom()`, drag management, and read-only `eye`/`lookAt`/`up` getters.

`OptiXCameraHandler` drops `SphericalOrbit` mixin and delegates to `OrbitCamera` instead.

The `CameraDragState` case class moves from `OptiXCameraHandler.scala` into this file.

### 4. `DragTracker.scala` — encapsulates shiftStart var

```scala
class DragTracker:
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var _origin: ScreenCoords = ScreenCoords(0, 0)

  def start(pos: ScreenCoords): Unit = _origin = pos
  def origin: ScreenCoords = _origin
```

---

## Files Modified

### `OptiXKeyHandler.scala`
- `var rotatePressed` → `val rotatePressed = KeyPressTracker()`
- 3 `scalafix:off/on null` blocks → `GdxRuntime.exit()` / check removed via zero-delta approach
- Remove `import com.badlogic.gdx.Gdx`, add `import menger.gdx.{GdxRuntime, KeyPressTracker}`

### `GdxKeyHandler.scala`
- `var rotatePressed` → `val rotatePressed = KeyPressTracker()`
- 1 `scalafix:off/on null` block → `GdxRuntime.exit()`

### `OptiXCameraHandler.scala` (largest change)
- Remove 5 vars (eye, lookAt, up, spherical, dragState)
- Remove `SphericalOrbit` mixin and all its delegated methods
- Remove `CameraDragState` case class
- Add `private val camera = OrbitCamera(initialEye, initialLookAt, initialUp, config)`
- Rewrite handlers to delegate: `handleOrbit` → `camera.orbit()`, etc.
- `Gdx.graphics.requestRendering()` → `GdxRuntime.requestRendering()`
- `Gdx.input.isKeyPressed()` → `GdxRuntime.isKeyPressed()`

### `GdxCameraHandler.scala`
- `var shiftStart` → `val dragTracker = DragTracker()`
- `Gdx.input.isKeyPressed/isButtonPressed` → `GdxRuntime.isKeyPressed/isButtonPressed` (optional, for consistency)

### `OptiXEngine.scala`
- `var currentObjectSpecs` → `AtomicReference` (follows existing `keyHandler` pattern on line 74)
- All `Gdx.app.exit()` → `GdxRuntime.exit()`
- All `Gdx.graphics.*` → `GdxRuntime.*`
- All `Gdx.input.*` → `GdxRuntime.*`
- All `Gdx.gl.*` → `GdxRuntime.*`

### `build.sbt`
- Add `menger\\.gdx\\..*` to `coverageExcludedPackages` (wrapper code is tested separately)

---

## Annotations Removed from Application Code

| Removed | Count |
|---------|-------|
| `@SuppressWarnings(Var)` | 9 |
| `scalafix:off/on null` pairs | 4 |

The 4 new wrapper files contain the minimum necessary suppressions (1 null pair in GdxRuntime, 7 Var annotations across 3 files).

---

## New Tests

| Test File | Tests | LibGDX needed? |
|-----------|-------|----------------|
| `GdxRuntimeSuite.scala` | `exit()`/`requestRendering()`/`deltaTime` safe when Gdx is null | No (nulls are the default) |
| `KeyPressTrackerSuite.scala` | press/release/isPressed/anyPressed, multi-key | No |
| `OrbitCameraSuite.scala` | orbit/pan/zoom change state, drag lifecycle, getters return copies | Yes (Vector3) |
| `DragTrackerSuite.scala` | start/origin, initial state | No |

---

## Implementation Order

1. Create `GdxRuntime.scala` + `GdxRuntimeSuite.scala`
2. Create `KeyPressTracker.scala` + `KeyPressTrackerSuite.scala`; update both key handlers
3. Create `DragTracker.scala` + `DragTrackerSuite.scala`; update `GdxCameraHandler`
4. Create `OrbitCamera.scala` + `OrbitCameraSuite.scala`; refactor `OptiXCameraHandler`
5. Refactor `OptiXEngine.currentObjectSpecs` → `AtomicReference`
6. Update `build.sbt` coverage exclusion
7. Verify: `sbt compile`, `sbt test`, `sbt "scalafix --check"`

**Commit:** `refactor: Introduce menger.gdx wrapper layer to eliminate var/null from application code`

---

## Verification

1. `sbt compile` — all types correct
2. `sbt test` — all ~1070 tests pass
3. `sbt "scalafix --check"` — no null/var violations outside `menger.gdx`
4. Manual grep:
   - `grep -r "private var" menger-app/src/main/scala/menger/` → only in `menger/gdx/` and `InputHandler.scala`
   - `grep -r "!= null" menger-app/src/main/scala/menger/` → zero results outside `menger/gdx/`
   - `grep -r "SuppressWarnings.*Var" menger-app/src/main/scala/menger/` → only in `menger/gdx/` and `InputHandler.scala`
