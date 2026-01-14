package menger.input

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.Input.{Buttons, Keys}
import com.badlogic.gdx.InputAdapter
import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.{OptiXRenderResources, RotationProjectionParameters}
import menger.optix.CameraState
import menger.optix.OptiXRendererWrapper

// Consolidated drag state - reduces 4 vars to 1 Option
private case class DragState(lastX: Int, lastY: Int, button: Int)

class OptiXCameraController(
  rendererWrapper: OptiXRendererWrapper,
  cameraState: CameraState,
  renderResources: OptiXRenderResources,
  initialEye: Vector3,
  initialLookAt: Vector3,
  initialUp: Vector3,
  dispatcher: EventDispatcher,
  config: OrbitConfig = OrbitConfig()
) extends InputAdapter with SphericalOrbit with LazyLogging:

  // SphericalOrbit config
  override protected def orbitConfig: OrbitConfig = config

  // 4D rotation sensitivity - convert pixel deltas to rotation degrees
  private final val rotation4DSensitivity = 0.3f

  // Camera position state - LibGDX Vector3 is inherently mutable
  // These vars are required for LibGDX integration which uses mutable vectors
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var eye: Vector3 = initialEye.cpy()

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var lookAt: Vector3 = initialLookAt.cpy()

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var up: Vector3 = initialUp.cpy()

  // Public getters for camera state (needed for scene rebuild)
  def currentEye: Vector3 = eye.cpy()
  def currentLookAt: Vector3 = lookAt.cpy()
  def currentUp: Vector3 = up.cpy()

  // Spherical coordinates - consolidated into single var
  private val (initAzimuth, initElevation, initDistance) = initSpherical(initialEye, initialLookAt)

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var spherical: SphericalCoords = SphericalCoords(initAzimuth, initElevation, initDistance)

  // SphericalOrbit trait implementation - delegate to consolidated state
  override protected def azimuth: Float = spherical.azimuth
  override protected def azimuth_=(value: Float): Unit =
    spherical = spherical.copy(azimuth = value)
  override protected def elevation: Float = spherical.elevation
  override protected def elevation_=(value: Float): Unit =
    spherical = spherical.copy(elevation = value)
  override protected def distance: Float = spherical.distance
  override protected def distance_=(value: Float): Unit =
    spherical = spherical.copy(distance = value)

  // Mouse tracking state - consolidated into Option[DragState]
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var dragState: Option[DragState] = None

  override def touchDown(screenX: Int, screenY: Int, pointer: Int, button: Int): Boolean =
    dragState = Some(DragState(screenX, screenY, button))
    true

  override def touchUp(screenX: Int, screenY: Int, pointer: Int, button: Int): Boolean =
    dragState = None
    true

  override def touchDragged(screenX: Int, screenY: Int, pointer: Int): Boolean =
    dragState match
      case None => false
      case Some(state) =>
        val deltaX = screenX - state.lastX
        val deltaY = screenY - state.lastY
        dragState = Some(state.copy(lastX = screenX, lastY = screenY))

        // Check if Shift key is pressed for 4D rotation
        val shiftPressed = Gdx.input.isKeyPressed(Keys.SHIFT_LEFT) || Gdx.input.isKeyPressed(Keys.SHIFT_RIGHT)

        if shiftPressed then
          handle4DRotation(state.button, deltaX, deltaY)
        else
          state.button match
            case Buttons.LEFT => handleOrbit(deltaX, deltaY)
            case Buttons.RIGHT => handlePan(deltaX, deltaY)
            case _ => // Ignore other buttons

        true

  override def scrolled(amountX: Float, amountY: Float): Boolean =
    handleZoom(amountY)
    true

  private def handleOrbit(deltaX: Int, deltaY: Int): Unit =
    updateOrbit(deltaX, deltaY)
    updateEyeFromSpherical()
    updateCamera()

  private def handlePan(deltaX: Int, deltaY: Int): Unit =
    val forward = lookAt.cpy().sub(eye).nor()
    val panOffset = computePanOffset(deltaX, deltaY, forward, up)
    eye.add(panOffset)
    lookAt.add(panOffset)
    updateCamera()

  private def handleZoom(scrollAmount: Float): Unit =
    updateZoom(scrollAmount)
    updateEyeFromSpherical()
    updateCamera()

  private def updateEyeFromSpherical(): Unit =
    val newEye = sphericalToCartesian(lookAt)
    eye.set(newEye.x, newEye.y, newEye.z)

  private def updateCamera(): Unit =
    cameraState.updateCamera(rendererWrapper.renderer, eye, lookAt, up)
    renderResources.markNeedsRender()
    Gdx.graphics.requestRendering()
    logger.debug(s"Camera updated: eye=$eye, lookAt=$lookAt, distance=$distance, azimuth=$azimuth, elevation=$elevation")

  private def handle4DRotation(button: Int, deltaX: Int, deltaY: Int): Unit =
    button match
      case Buttons.LEFT =>
        // Left drag: horizontal controls XW, vertical controls YW
        val rotXW = -deltaX * rotation4DSensitivity  // Negative for intuitive direction
        val rotYW = deltaY * rotation4DSensitivity   // Positive matches arrow key convention
        logger.debug(s"Shift + left drag: deltaX=$deltaX, deltaY=$deltaY -> rotXW=$rotXW, rotYW=$rotYW")
        dispatcher.notifyObservers(RotationProjectionParameters(rotXW, rotYW, 0f))
      case Buttons.RIGHT =>
        // Right drag: vertical controls ZW
        val rotZW = deltaY * rotation4DSensitivity
        logger.debug(s"Shift + right drag: deltaY=$deltaY -> rotZW=$rotZW")
        dispatcher.notifyObservers(RotationProjectionParameters(0f, 0f, rotZW))
      case _ =>
        // Ignore other buttons
