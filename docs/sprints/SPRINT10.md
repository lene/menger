# Sprint 10: 4D Framework

**Sprint:** 10 - 4D Framework
**Status:** Not Started
**Estimate:** 20-25 hours
**Branch:** `feature/sprint-10`
**Depends on:** Sprint 8 (TesseractMesh), Sprint 9 (TesseractSponge)

---

## Goal

Enable interactive 4D manipulation in OptiX mode, with enhanced CLI parameters and a clean abstract 4D API for extensibility. Support **multiple tesseract instances with independent 4D rotations** in the same scene.

## Success Criteria

- [ ] Interactive 4D rotation in OptiX mode (Shift+Arrow keys, Shift+Mouse drag)
- [ ] Interactive 4D projection adjustment (Shift+Scroll changes `eyeW`)
- [ ] Reset 4D view to defaults (ESC key in OptiX mode)
- [ ] Per-instance 4D parameters in multi-object scenes
- [ ] CLI: `--4d-rotation=XW,YW,ZW` shorthand for 4D rotation angles
- [ ] CLI: `--4d-preset=NAME` for common 4D views (edge-on, face-on, cell-on)
- [ ] State persistence: save/load 4D view parameters
- [ ] Abstract `Mesh4D` interface formalized for future extensions
- [ ] All tests pass (~30-40 new tests)

---

## Background

### Current State (after Sprints 8 & 9)

| Component | Status |
|-----------|--------|
| 4D math (`Rotation`, `Projection`, `Vector[4]`, `Matrix[4]`) | ✅ Complete |
| 4D objects (`Tesseract`, `TesseractSponge`, `TesseractSponge2`) | ✅ Complete |
| LibGDX 4D interaction (Shift+Arrow/Mouse) | ✅ Complete |
| OptiX 4D rendering (`TesseractMesh`, `Mesh4DProjection`) | ✅ Complete (Sprint 8-9) |
| OptiX 4D interaction | ❌ Not implemented |
| Per-instance 4D parameters | ❌ All instances share same projection |
| 4D CLI presets | ❌ Not implemented |

### Key Challenge: Re-rendering on 4D Rotation

Unlike 3D camera movement (which only updates OptiX camera parameters), 4D rotation requires **regenerating the mesh** because:
- 4D→3D projection happens on the CPU
- The projected 3D mesh changes with each 4D rotation
- OptiX sees a different triangle mesh each frame

**Architecture Decision:** Accept mesh regeneration cost for now. 4D rotations are infrequent user interactions, not continuous animations. Future optimization (shader-based 4D) deferred to backlog.

---

## Tasks

### Step 10.1: Create OptiX4DController Class

**Status:** Not Started
**Estimate:** 3 hours

Create a new controller class that handles 4D rotation/projection input in OptiX mode.

#### Subtasks

- [ ] Create `OptiX4DController` class implementing `Observer`
- [ ] Handle Shift+Arrow keys for 4D rotation (XW, YW, ZW planes)
- [ ] Handle Shift+Scroll for projection adjustment (eyeW)
- [ ] Create `EventDispatcher` instance for OptiX mode
- [ ] Track current 4D rotation state
- [ ] Provide callback mechanism to notify engine of 4D changes

#### Files to Create

**`menger-app/src/main/scala/menger/input/OptiX4DController.scala`**

```scala
package menger.input

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.Input.Keys
import com.typesafe.scalalogging.LazyLogging
import menger.RotationProjectionParameters
import menger.common.Const

/**
 * Handles 4D rotation and projection input for OptiX mode.
 *
 * Controls:
 * - Shift + Left/Right: Rotate in XW plane
 * - Shift + Up/Down: Rotate in YW plane
 * - Shift + PageUp/PageDown: Rotate in ZW plane
 * - Shift + Scroll: Adjust eyeW (4D projection distance)
 * - ESC: Reset to default 4D view
 */
class OptiX4DController(
  initialParams: RotationProjectionParameters,
  onChange: RotationProjectionParameters => Unit
) extends LazyLogging:

  // Current 4D state - accumulates rotation/projection changes
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var currentParams: RotationProjectionParameters = initialParams

  private val defaultParams: RotationProjectionParameters = initialParams

  private final val rotateAngle = 45f  // degrees per second

  def params: RotationProjectionParameters = currentParams

  def update(): Unit =
    if isShiftPressed && anyRotationKeyPressed then
      val delta = Gdx.graphics.getDeltaTime
      applyRotation(delta)

  private def isShiftPressed: Boolean =
    Gdx.input.isKeyPressed(Keys.SHIFT_LEFT) || Gdx.input.isKeyPressed(Keys.SHIFT_RIGHT)

  private def anyRotationKeyPressed: Boolean =
    Seq(Keys.LEFT, Keys.RIGHT, Keys.UP, Keys.DOWN, Keys.PAGE_UP, Keys.PAGE_DOWN)
      .exists(Gdx.input.isKeyPressed)

  private def applyRotation(delta: Float): Unit =
    val dXW = angle(delta, Keys.LEFT, Keys.RIGHT)
    val dYW = angle(delta, Keys.UP, Keys.DOWN)
    val dZW = angle(delta, Keys.PAGE_UP, Keys.PAGE_DOWN)

    if dXW != 0f || dYW != 0f || dZW != 0f then
      val deltaParams = RotationProjectionParameters(dXW, dYW, dZW)
      currentParams = currentParams + deltaParams
      logger.debug(s"4D rotation: XW=${currentParams.rotXW}, YW=${currentParams.rotYW}, ZW=${currentParams.rotZW}")
      onChange(currentParams)

  private def angle(delta: Float, negKey: Int, posKey: Int): Float =
    val neg = if Gdx.input.isKeyPressed(negKey) then -1 else 0
    val pos = if Gdx.input.isKeyPressed(posKey) then 1 else 0
    delta * rotateAngle * (neg + pos)

  def applyDelta(dXW: Float, dYW: Float, dZW: Float): Unit =
    if dXW != 0f || dYW != 0f || dZW != 0f then
      val deltaParams = RotationProjectionParameters(dXW, dYW, dZW)
      currentParams = currentParams + deltaParams
      onChange(currentParams)

  def handleScroll(amountY: Float): Boolean =
    if isShiftPressed then
      // Exponential scaling for smooth zoom feel
      val factor = Math.pow(1.1, amountY.toDouble).toFloat
      val newEyeW = (currentParams.eyeW * factor).max(currentParams.screenW + 0.1f)
      currentParams = currentParams.copy(eyeW = newEyeW)
      logger.debug(s"4D projection: eyeW=${currentParams.eyeW}")
      onChange(currentParams)
      true
    else
      false

  def resetToDefault(): Unit =
    currentParams = defaultParams
    logger.info("Reset 4D view to defaults")
    onChange(currentParams)
```

#### Tests to Add

**`menger-app/src/test/scala/menger/input/OptiX4DControllerSpec.scala`**

```scala
package menger.input

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.RotationProjectionParameters

class OptiX4DControllerSpec extends AnyFlatSpec with Matchers:

  "OptiX4DController" should "start with initial parameters" in:
    val initial = RotationProjectionParameters(15f, 10f, 0f)
    var callbackInvoked = false
    val controller = OptiX4DController(initial, _ => callbackInvoked = true)

    controller.params shouldBe initial

  it should "accumulate rotation changes via applyDelta" in:
    val initial = RotationProjectionParameters(0f, 0f, 0f)
    var lastParams: RotationProjectionParameters = initial
    val controller = OptiX4DController(initial, p => lastParams = p)

    controller.applyDelta(10f, 20f, 30f)
    lastParams.rotXW shouldBe 10f
    lastParams.rotYW shouldBe 20f
    lastParams.rotZW shouldBe 30f

    controller.applyDelta(5f, -10f, 15f)
    lastParams.rotXW shouldBe 15f
    lastParams.rotYW shouldBe 10f
    lastParams.rotZW shouldBe 45f

  it should "reset to defaults" in:
    val initial = RotationProjectionParameters(15f, 10f, 5f)
    var lastParams = initial
    val controller = OptiX4DController(initial, p => lastParams = p)

    controller.applyDelta(30f, 40f, 50f)
    controller.resetToDefault()
    lastParams shouldBe initial

  it should "clamp eyeW above screenW" in:
    val initial = RotationProjectionParameters(0f, 0f, 0f, eyeW = 3f, screenW = 1.5f)
    var lastParams = initial
    val controller = OptiX4DController(initial, p => lastParams = p)

    // eyeW should never go below screenW + epsilon
    controller.params.eyeW should be > controller.params.screenW
```

---

### Step 10.2: Integrate OptiX4DController into OptiXKeyController

**Status:** Not Started
**Estimate:** 2 hours

Extend `OptiXKeyController` to delegate 4D input to `OptiX4DController`.

#### Subtasks

- [ ] Add `OptiX4DController` dependency to `OptiXKeyController`
- [ ] Call `controller.update()` on each frame
- [ ] Handle ESC key to reset 4D view (in addition to existing behavior)
- [ ] Update `BaseKeyController` to track PageUp/PageDown keys

#### Files to Modify

**`menger-app/src/main/scala/menger/input/BaseKeyController.scala`**

Add PageUp/PageDown to tracked keys:
```scala
// Add to rotatePressed map initialization
Keys.PAGE_UP -> false, Keys.PAGE_DOWN -> false
```

**`menger-app/src/main/scala/menger/input/OptiXKeyController.scala`**

```scala
package menger.input

import com.badlogic.gdx.Gdx
import menger.RotationProjectionParameters

class OptiXKeyController(
  fourDController: Option[OptiX4DController] = None
) extends BaseKeyController:

  override def keyDown(keycode: Int): Boolean =
    val handled = super.keyDown(keycode)
    // Additional handling can go here
    handled

  override protected def handleEscape(): Boolean =
    // Reset 4D view if controller exists, then exit
    fourDController.foreach(_.resetToDefault())
    Gdx.app.exit()
    true

  override protected def onRotationUpdate(): Unit =
    fourDController.foreach(_.update())
```

---

### Step 10.3: Enable 4D Mesh Regeneration in OptiXEngine

**Status:** Not Started
**Estimate:** 4 hours

Add the ability for `OptiXEngine` to regenerate 4D meshes when rotation parameters change.

#### Subtasks

- [ ] Store current `RotationProjectionParameters` in engine state
- [ ] Create method to regenerate 4D mesh with new parameters
- [ ] Update mesh in OptiX renderer after regeneration
- [ ] Track whether scene contains 4D objects (to skip regeneration for 3D-only scenes)
- [ ] Request re-render after mesh update

#### Files to Modify

**`menger-app/src/main/scala/menger/engines/OptiXEngine.scala`**

Add 4D state management:
```scala
// New imports
import menger.RotationProjectionParameters
import menger.input.OptiX4DController
import menger.objects.higher_d.TesseractMesh
import menger.objects.higher_d.TesseractSpongeMesh
import menger.objects.higher_d.TesseractSponge2Mesh

// Add to class body:

// 4D state - only used when scene contains 4D objects
@SuppressWarnings(Array("org.wartremover.warts.Var"))
private var current4DParams: Option[RotationProjectionParameters] = None

private lazy val fourDController: Option[OptiX4DController] =
  if has4DObjects then
    val initial = RotationProjectionParameters(
      rotXW = default4DRotXW,
      rotYW = default4DRotYW,
      rotZW = default4DRotZW,
      eyeW = default4DEyeW,
      screenW = default4DScreenW
    )
    current4DParams = Some(initial)
    Some(OptiX4DController(initial, on4DParamsChanged))
  else
    None

private def has4DObjects: Boolean =
  scene.objectSpecs.exists(_.exists(spec => ObjectType.isHypercube(spec.objectType))) ||
  ObjectType.isHypercube(scene.spongeType)

private def on4DParamsChanged(params: RotationProjectionParameters): Unit =
  current4DParams = Some(params)
  regenerate4DMeshes(params)
  renderResources.markNeedsRender()
  Gdx.graphics.requestRendering()

private def regenerate4DMeshes(params: RotationProjectionParameters): Unit =
  scene.objectSpecs match
    case Some(specs) =>
      // Multi-object mode: regenerate all 4D meshes
      val fourDSpecs = specs.filter(s => ObjectType.isHypercube(s.objectType))
      if fourDSpecs.nonEmpty then
        regenerateMultiObject4DMeshes(fourDSpecs, params)
    case None =>
      // Single-object mode
      if ObjectType.isHypercube(scene.spongeType) then
        regenerateSingleObject4DMesh(params)

private def regenerateSingleObject4DMesh(params: RotationProjectionParameters): Unit =
  val mesh = create4DMeshWithParams(scene.spongeType, scene.sphereRadius * 2, scene.level, params)
  rendererWrapper.renderer.setTriangleMesh(mesh)

private def create4DMeshWithParams(
  objectType: String,
  size: Float,
  level: Float,
  params: RotationProjectionParameters
): menger.common.TriangleMeshData =
  objectType match
    case "tesseract" =>
      TesseractMesh(
        size = size,
        eyeW = params.eyeW,
        screenW = params.screenW,
        rotXW = params.rotXW,
        rotYW = params.rotYW,
        rotZW = params.rotZW
      ).toTriangleMesh
    case "tesseract-sponge" =>
      TesseractSpongeMesh(
        size = size,
        level = level,
        eyeW = params.eyeW,
        screenW = params.screenW,
        rotXW = params.rotXW,
        rotYW = params.rotYW,
        rotZW = params.rotZW
      ).toTriangleMesh
    case "tesseract-sponge-2" =>
      TesseractSponge2Mesh(
        size = size,
        level = level,
        eyeW = params.eyeW,
        screenW = params.screenW,
        rotXW = params.rotXW,
        rotYW = params.rotYW,
        rotZW = params.rotZW
      ).toTriangleMesh
    case other =>
      throw IllegalArgumentException(s"Not a 4D object type: $other")
```

---

### Step 10.4: Add 4D Mouse Controls to OptiXCameraController

**Status:** Not Started
**Estimate:** 2 hours

Extend `OptiXCameraController` to handle Shift+Mouse for 4D rotation and Shift+Scroll for projection.

#### Subtasks

- [ ] Pass `OptiX4DController` reference to `OptiXCameraController`
- [ ] Detect Shift+Drag and route to 4D controller
- [ ] Detect Shift+Scroll and route to 4D controller
- [ ] Preserve existing 3D camera behavior when Shift is not pressed

#### Files to Modify

**`menger-app/src/main/scala/menger/input/OptiXCameraController.scala`**

```scala
// Add parameter:
class OptiXCameraController(
  rendererWrapper: OptiXRendererWrapper,
  cameraState: CameraState,
  renderResources: OptiXRenderResources,
  initialEye: Vector3,
  initialLookAt: Vector3,
  initialUp: Vector3,
  config: OrbitConfig = OrbitConfig(),
  fourDController: Option[OptiX4DController] = None  // NEW
) extends InputAdapter with SphericalOrbit with LazyLogging:

  // ... existing code ...

  override def touchDragged(screenX: Int, screenY: Int, pointer: Int): Boolean =
    dragState match
      case None => false
      case Some(state) =>
        val deltaX = screenX - state.lastX
        val deltaY = screenY - state.lastY
        dragState = Some(state.copy(lastX = screenX, lastY = screenY))

        if isShiftPressed then
          // 4D rotation via mouse drag
          fourDController.foreach { ctrl =>
            handle4DDrag(deltaX, deltaY, state.button, ctrl)
          }
        else
          state.button match
            case Buttons.LEFT => handleOrbit(deltaX, deltaY)
            case Buttons.RIGHT => handlePan(deltaX, deltaY)
            case _ => // Ignore other buttons

        true

  private def handle4DDrag(deltaX: Int, deltaY: Int, button: Int, ctrl: OptiX4DController): Unit =
    val sensitivity = 0.5f
    button match
      case Buttons.LEFT =>
        // Left drag: XW and YW rotation
        val dXW = deltaX * sensitivity
        val dYW = -deltaY * sensitivity
        ctrl.applyDelta(dXW, dYW, 0f)
      case Buttons.RIGHT =>
        // Right drag: ZW rotation
        val dZW = deltaX * sensitivity
        ctrl.applyDelta(0f, 0f, dZW)
      case _ => ()

  override def scrolled(amountX: Float, amountY: Float): Boolean =
    if isShiftPressed then
      fourDController.map(_.handleScroll(amountY)).getOrElse(false)
    else
      handleZoom(amountY)
      true

  private def isShiftPressed: Boolean =
    Gdx.input.isKeyPressed(Keys.SHIFT_LEFT) || Gdx.input.isKeyPressed(Keys.SHIFT_RIGHT)
```

---

### Step 10.5: Add 4D CLI Enhancements

**Status:** Not Started
**Estimate:** 2.5 hours

Add new CLI options for convenient 4D parameter specification.

#### Subtasks

- [ ] Add `--4d-rotation=XW,YW,ZW` shorthand option
- [ ] Add `--4d-preset=NAME` for common views
- [ ] Add preset definitions (edge-on, face-on, cell-on, classic)
- [ ] Validate that shorthand and explicit options don't conflict
- [ ] Update help text

#### Files to Modify

**`menger-app/src/main/scala/menger/MengerCLIOptions.scala`**

```scala
// Add new options in projection group:

val fourDRotation: ScallopOption[String] = opt[String](
  name = "4d-rotation", required = false, group = projectionGroup,
  descr = "4D rotation angles as XW,YW,ZW in degrees (e.g., --4d-rotation=30,20,0)"
)

val fourDPreset: ScallopOption[String] = opt[String](
  name = "4d-preset", required = false, group = projectionGroup,
  descr = "4D view preset: classic (default), edge-on, face-on, cell-on, flat"
)

// Add validation:
validateOpt(fourDRotation, rotXW, rotYW, rotZW) { (rotation, xw, yw, zw) =>
  if rotation.isDefined && (xw.isDefined || yw.isDefined || zw.isDefined) then
    Left("Cannot specify both --4d-rotation and individual --rot-xw/--rot-yw/--rot-zw options")
  else
    Right(())
}
```

#### Files to Create

**`menger-app/src/main/scala/menger/config/FourDPresets.scala`**

```scala
package menger.config

import menger.RotationProjectionParameters
import menger.common.Const

object FourDPresets:

  val presets: Map[String, RotationProjectionParameters] = Map(
    "classic" -> RotationProjectionParameters(
      rotXW = 15f, rotYW = 10f, rotZW = 0f,
      eyeW = Const.defaultEyeW, screenW = Const.defaultScreenW
    ),
    "edge-on" -> RotationProjectionParameters(
      rotXW = 45f, rotYW = 0f, rotZW = 0f,
      eyeW = 3f, screenW = 1.5f
    ),
    "face-on" -> RotationProjectionParameters(
      rotXW = 0f, rotYW = 0f, rotZW = 0f,
      eyeW = 3f, screenW = 1.5f
    ),
    "cell-on" -> RotationProjectionParameters(
      rotXW = 45f, rotYW = 45f, rotZW = 0f,
      eyeW = 3f, screenW = 1.5f
    ),
    "flat" -> RotationProjectionParameters(
      rotXW = 0f, rotYW = 0f, rotZW = 0f,
      eyeW = 10f, screenW = 9f  // Nearly orthographic projection
    )
  )

  def get(name: String): Option[RotationProjectionParameters] =
    presets.get(name.toLowerCase)

  def names: Seq[String] = presets.keys.toSeq.sorted
```

---

### Step 10.6: Per-Instance 4D Parameters in Multi-Object Mode

**Status:** Not Started
**Estimate:** 3 hours

Enable different 4D rotations for different tesseract instances in the same scene.

#### Subtasks

- [ ] Extend `ObjectSpec` to include 4D parameters (if not done in Sprint 8)
- [ ] Store per-instance 4D params alongside mesh data
- [ ] Generate separate meshes for instances with different 4D rotations
- [ ] Group instances by 4D parameters for efficient rendering

#### Architecture Decision

Two approaches:
1. **Generate separate meshes** for each unique 4D parameter set (current approach)
2. **Share mesh, use shader-time 4D rotation** (future optimization)

Sprint 10 uses approach 1 for simplicity. Instances with identical 4D params share a mesh.

#### Files to Modify

**`menger-app/src/main/scala/menger/engines/OptiXEngine.scala`**

Update `setupMultipleTriangleMeshes` to handle 4D objects:

```scala
private def setupMultipleTriangleMeshes(specs: List[ObjectSpec], renderer: OptiXRenderer): Try[Unit] = Try:
  // Group specs by mesh identity (type + level + 4D params for hypercubes)
  val meshGroups = groupSpecsByMeshIdentity(specs)

  meshGroups.foreach { case (meshKey, groupSpecs) =>
    val firstSpec = groupSpecs.head
    val mesh = createMeshForSpec(firstSpec)

    // If this is the first mesh, set it as the base
    // Otherwise, we need multi-mesh support (future enhancement)
    if meshKey == meshGroups.keys.head then
      renderer.setTriangleMesh(mesh)

    // Add instances for this mesh group
    groupSpecs.foreach { spec =>
      val position = menger.common.Vector[3](spec.x, spec.y, spec.z)
      val (color, ior) = extractMaterialProperties(spec)
      renderer.addTriangleMeshInstance(position, color, ior, -1)
    }
  }

private case class MeshKey(
  objectType: String,
  level: Option[Float],
  rotXW: Float,
  rotYW: Float,
  rotZW: Float,
  eyeW: Float,
  screenW: Float
)

private def groupSpecsByMeshIdentity(specs: List[ObjectSpec]): Map[MeshKey, List[ObjectSpec]] =
  specs.groupBy { spec =>
    MeshKey(
      spec.objectType,
      spec.level,
      spec.rotXW,
      spec.rotYW,
      spec.rotZW,
      spec.eyeW,
      spec.screenW
    )
  }
```

---

### Step 10.7: Abstract Mesh4D Interface Formalization

**Status:** Not Started
**Estimate:** 2 hours

Clean up and document the 4D mesh interface for future extensibility.

#### Subtasks

- [ ] Add documentation to `Mesh4D` trait
- [ ] Add `Fractal4D` documentation
- [ ] Create `Mesh4DSource` trait (analogous to `TriangleMeshSource`)
- [ ] Ensure consistent API across all 4D types
- [ ] Add tests for interface compliance

#### Files to Modify

**`menger-app/src/main/scala/menger/objects/higher_d/Mesh4D.scala`**

```scala
package menger.objects.higher_d

/**
 * Trait for objects that represent 4D mesh geometry.
 *
 * Implementors provide a sequence of 4D faces (quadrilaterals in 4D space)
 * that can be projected to 3D for rendering.
 *
 * Known implementors:
 * - Tesseract: Basic 4D hypercube (24 faces)
 * - TesseractSponge: Volume-based 4D Menger analog
 * - TesseractSponge2: Surface-based 4D fractal
 */
trait Mesh4D:
  /**
   * All faces of this 4D mesh.
   * Each face is a quadrilateral defined by 4 vertices in 4D space.
   */
  def faces: Seq[Face4D]

  /** Number of faces in this mesh */
  def faceCount: Int = faces.size

  /** Number of vertices (faces * 4, may include duplicates) */
  def vertexCount: Int = faces.size * 4
```

**`menger-app/src/main/scala/menger/objects/higher_d/Fractal4D.scala`**

```scala
package menger.objects.higher_d

/**
 * Trait for 4D fractal objects with a recursion level.
 *
 * Extends Mesh4D with level-based geometry generation.
 * Supports fractional levels for smooth animation transitions.
 */
trait Fractal4D extends Mesh4D:
  /**
   * Fractal recursion level.
   * - Level 0: Base shape (e.g., Tesseract)
   * - Level N: N iterations of subdivision
   * - Fractional levels (e.g., 1.5): Interpolated geometry
   */
  def level: Float
```

---

### Step 10.8: State Persistence (Save/Load 4D View)

**Status:** Not Started
**Estimate:** 2 hours

Add ability to save and restore 4D view parameters.

#### Subtasks

- [ ] Define JSON format for 4D state
- [ ] Add `--save-4d-state=FILE` CLI option
- [ ] Add `--load-4d-state=FILE` CLI option
- [ ] Save state on exit (optional: `--auto-save-4d`)
- [ ] Add tests for serialization/deserialization

#### Files to Create

**`menger-app/src/main/scala/menger/config/FourDState.scala`**

```scala
package menger.config

import scala.util.Try

import menger.RotationProjectionParameters

/**
 * Serializable 4D view state for save/load functionality.
 */
case class FourDState(
  rotXW: Float,
  rotYW: Float,
  rotZW: Float,
  eyeW: Float,
  screenW: Float
):
  def toParams: RotationProjectionParameters =
    RotationProjectionParameters(rotXW, rotYW, rotZW, eyeW, screenW)

object FourDState:
  def fromParams(params: RotationProjectionParameters): FourDState =
    FourDState(params.rotXW, params.rotYW, params.rotZW, params.eyeW, params.screenW)

  def toJson(state: FourDState): String =
    s"""{
       |  "rotXW": ${state.rotXW},
       |  "rotYW": ${state.rotYW},
       |  "rotZW": ${state.rotZW},
       |  "eyeW": ${state.eyeW},
       |  "screenW": ${state.screenW}
       |}""".stripMargin

  def fromJson(json: String): Try[FourDState] = Try:
    // Simple JSON parsing (or use a library like circe/upickle)
    val rotXW = extractFloat(json, "rotXW")
    val rotYW = extractFloat(json, "rotYW")
    val rotZW = extractFloat(json, "rotZW")
    val eyeW = extractFloat(json, "eyeW")
    val screenW = extractFloat(json, "screenW")
    FourDState(rotXW, rotYW, rotZW, eyeW, screenW)

  private def extractFloat(json: String, key: String): Float =
    val pattern = s""""$key":\\s*([\\d.\\-]+)""".r
    pattern.findFirstMatchIn(json).map(_.group(1).toFloat).getOrElse(0f)

  def save(state: FourDState, path: String): Try[Unit] = Try:
    val json = toJson(state)
    java.nio.file.Files.writeString(java.nio.file.Path.of(path), json)

  def load(path: String): Try[FourDState] =
    Try(java.nio.file.Files.readString(java.nio.file.Path.of(path))).flatMap(fromJson)
```

---

### Step 10.9: Integration Tests and Manual Verification

**Status:** Not Started
**Estimate:** 3 hours

Create integration tests and perform manual visual verification.

#### Subtasks

- [ ] Create integration test file
- [ ] Test 4D rotation produces different meshes
- [ ] Test 4D presets generate expected geometry
- [ ] Test multi-instance with different 4D params
- [ ] Manual verification commands
- [ ] Take screenshots for documentation

#### Files to Create

**`menger-app/src/test/scala/menger/engines/OptiX4DIntegrationSpec.scala`**

```scala
package menger.engines

import menger.ObjectSpec
import menger.RotationProjectionParameters
import menger.config.FourDPresets
import menger.objects.higher_d.TesseractMesh
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class OptiX4DIntegrationSpec extends AnyFlatSpec with Matchers:

  "4D presets" should "all be defined" in:
    FourDPresets.names should contain allOf ("classic", "edge-on", "face-on", "cell-on", "flat")

  it should "return valid parameters for each preset" in:
    FourDPresets.names.foreach { name =>
      val params = FourDPresets.get(name)
      params shouldBe defined
      params.get.eyeW should be > params.get.screenW
    }

  "Different 4D rotations" should "produce different meshes" in:
    val mesh1 = TesseractMesh(rotXW = 0f, rotYW = 0f).toTriangleMesh
    val mesh2 = TesseractMesh(rotXW = 45f, rotYW = 0f).toTriangleMesh

    mesh1.vertices should not equal mesh2.vertices

  "Different eyeW values" should "produce different mesh sizes" in:
    val near = TesseractMesh(eyeW = 3f, screenW = 1.5f).toTriangleMesh
    val far = TesseractMesh(eyeW = 10f, screenW = 1.5f).toTriangleMesh

    def boundingBox(mesh: menger.common.TriangleMeshData): Float =
      val xs = (0 until mesh.numVertices).map(i => mesh.vertices(i * 8))
      xs.max - xs.min

    boundingBox(near) should be > boundingBox(far)

  "RotationProjectionParameters" should "accumulate correctly" in:
    val p1 = RotationProjectionParameters(10f, 20f, 30f)
    val p2 = RotationProjectionParameters(5f, -10f, 15f)
    val sum = p1 + p2

    sum.rotXW shouldBe 15f
    sum.rotYW shouldBe 10f
    sum.rotZW shouldBe 45f
```

#### Manual Verification Commands

```bash
# Interactive 4D rotation (requires GPU)
sbt "run --optix --objects type=tesseract:color=#4488FF --timeout 30"
# Use Shift+Arrow keys to rotate in 4D
# Use Shift+Scroll to adjust projection distance

# 4D presets
sbt "run --optix --objects type=tesseract:color=#FF8844 --4d-preset=edge-on --save-name tesseract-edge-on.png --timeout 3"
sbt "run --optix --objects type=tesseract:color=#44FF88 --4d-preset=cell-on --save-name tesseract-cell-on.png --timeout 3"
sbt "run --optix --objects type=tesseract:color=#FF44FF --4d-preset=flat --save-name tesseract-flat.png --timeout 3"

# 4D rotation shorthand
sbt "run --optix --objects type=tesseract:color=#FFFF44 --4d-rotation=45,30,15 --save-name tesseract-custom-rotation.png --timeout 3"

# Multiple tesseracts with different 4D rotations
sbt "run --optix --objects type=tesseract:pos=-2,0,0:rot-xw=0:rot-yw=0:color=#FF0000 --objects type=tesseract:pos=2,0,0:rot-xw=45:rot-yw=30:color=#00FF00 --save-name tesseract-multi-rotation.png --timeout 5"

# 4D sponge with interactive rotation
sbt "run --optix --objects type=tesseract-sponge:level=1:color=#8888FF --timeout 30"
```

---

### Step 10.10: Update Documentation

**Status:** Not Started
**Estimate:** 1 hour

Update changelog, roadmap, and documentation.

#### Subtasks

- [ ] Update CHANGELOG.md
- [ ] Update ROADMAP.md to mark Sprint 10 complete
- [ ] Update CLI help text
- [ ] Archive sprint documentation

#### Files to Modify

**`CHANGELOG.md`** (add after Sprint 9 entry):

```markdown
## [0.5.0] - 2026-XX-XX

### Added
- **Tesseract (4D Hypercube)** - Render 4D geometry projected to 3D via OptiX
  - `--objects type=tesseract` for 4D hypercube rendering
  - 4D projection parameters: `eye-w`, `screen-w`
  - 4D rotation parameters: `rot-xw`, `rot-yw`, `rot-zw`

- **TesseractSponge (4D Menger Sponge)** - Render 4D fractal sponges
  - `--objects type=tesseract-sponge:level=N` - Volume-based 4D sponge (levels 0-3)
  - `--objects type=tesseract-sponge-2:level=N` - Surface-based 4D sponge (levels 0-4)
  - Fractional level support for smooth animations
  - Full material support (glass, chrome, etc.)

- **Interactive 4D Manipulation in OptiX Mode**
  - Shift + Arrow keys: Rotate in XW/YW/ZW planes
  - Shift + Mouse drag: XW/YW rotation (left), ZW rotation (right)
  - Shift + Scroll: Adjust 4D projection distance (eyeW)
  - ESC: Reset 4D view to defaults

- **4D CLI Enhancements**
  - `--4d-rotation=XW,YW,ZW` shorthand for rotation angles
  - `--4d-preset=NAME` for common views (classic, edge-on, face-on, cell-on, flat)
  - `--save-4d-state=FILE` / `--load-4d-state=FILE` for view persistence

- **Per-Instance 4D Parameters**
  - Different tesseracts in the same scene can have different 4D rotations
  - Example: `--objects type=tesseract:pos=-2,0,0:rot-xw=0 --objects type=tesseract:pos=2,0,0:rot-xw=45`

### Changed
- Refactored `TesseractMesh` to `Mesh4DProjection` for reuse with any 4D geometry
- Abstract `Mesh4D` and `Fractal4D` interfaces documented for extensibility
```

**`ROADMAP.md`** - Update completed sprints table:

```markdown
| 8 | 4D Projection Foundation | ✅ Complete | [archive](docs/archive/sprints/) |
| 9 | TesseractSponge | ✅ Complete | [archive](docs/archive/sprints/) |
| 10 | 4D Framework | ✅ Complete | [archive](docs/archive/sprints/) |
```

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing (Scala + C++): `sbt test --warn`
- [ ] Code compiles without warnings: `sbt compile`
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] Manual verification screenshots captured
- [ ] Sprint documentation archived
- [ ] Interactive 4D demo works on GPU machine

---

## Summary

| Step | Task | Estimate | Files |
|------|------|----------|-------|
| 10.1 | Create OptiX4DController | 3h | New: `OptiX4DController.scala` |
| 10.2 | Integrate into OptiXKeyController | 2h | `OptiXKeyController.scala`, `BaseKeyController.scala` |
| 10.3 | Enable 4D mesh regeneration | 4h | `OptiXEngine.scala` |
| 10.4 | Add 4D mouse controls | 2h | `OptiXCameraController.scala` |
| 10.5 | Add 4D CLI enhancements | 2.5h | `MengerCLIOptions.scala`, new: `FourDPresets.scala` |
| 10.6 | Per-instance 4D parameters | 3h | `OptiXEngine.scala` |
| 10.7 | Abstract Mesh4D interface | 2h | `Mesh4D.scala`, `Fractal4D.scala` |
| 10.8 | State persistence | 2h | New: `FourDState.scala`, `MengerCLIOptions.scala` |
| 10.9 | Integration tests | 3h | New: `OptiX4DIntegrationSpec.scala` |
| 10.10 | Update documentation | 1h | `CHANGELOG.md`, `ROADMAP.md` |
| **Total** | | **24.5h** | |

---

## Notes

### Decisions Made

1. **Mesh regeneration on 4D rotation:** Accept CPU cost for now; shader-based 4D rotation deferred
2. **Per-instance 4D params:** Group instances by 4D params to minimize mesh duplication
3. **4D presets:** Five common views: classic, edge-on, face-on, cell-on, flat
4. **State format:** Simple JSON for portability

### Dependencies on Sprints 8 & 9

This sprint assumes:
- `ObjectSpec` has 4D fields (`eyeW`, `screenW`, `rotXW`, `rotYW`, `rotZW`)
- `TesseractMesh`, `TesseractSpongeMesh`, `TesseractSponge2Mesh` exist
- `ObjectType.isHypercube()` recognizes all 4D types
- `createMeshForSpec()` handles tesseract types

### Potential Issues

1. **Performance:** High-level 4D sponges may lag during interactive rotation (millions of triangles)
2. **Multi-mesh IAS:** Current IAS assumes single mesh type; per-instance 4D may need multi-mesh GAS
3. **Input focus:** LibGDX input may conflict with other UI elements

### Future Enhancements (Backlog)

- Shader-time 4D rotation (eliminates mesh regeneration)
- 4D animation timeline
- 4D cross-section (slice through W)
- VR support for true 4D immersion

---

## References

- Existing 4D code: `menger-app/src/main/scala/menger/objects/higher_d/`
- LibGDX 4D controls: `GdxKeyController.scala`, `GdxCameraController.scala`
- OptiX integration: `menger-app/src/main/scala/menger/engines/OptiXEngine.scala`
- Sprint 8 plan: `SPRINT8.md`
- Sprint 9 plan: `SPRINT9.md`
