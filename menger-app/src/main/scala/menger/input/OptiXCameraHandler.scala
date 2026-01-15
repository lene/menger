package menger.input

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.Input.Keys
import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.OptiXRenderResources
import menger.RotationProjectionParameters
import menger.common.Const
import menger.common.MouseButton
import menger.common.ScreenCoords
import menger.optix.CameraState
import menger.optix.OptiXRendererWrapper

/** Consolidated drag state - reduces multiple vars to single Option */
private case class CameraDragState(lastPos: ScreenCoords, button: MouseButton)

/**
 * Camera/mouse input handler for OptiX ray-traced rendering mode.
 *
 * Implements spherical orbit camera system with support for:
 * - Left drag: orbit camera (azimuth/elevation)
 * - Right drag: pan camera
 * - Scroll: zoom (change distance)
 * - Shift + left drag: 4D XW/YW rotation
 * - Shift + right drag: 4D ZW rotation
 *
 * Uses SphericalOrbit trait for camera math (azimuth, elevation, distance).
 */
class OptiXCameraHandler(
  rendererWrapper: OptiXRendererWrapper,
  cameraState: CameraState,
  renderResources: OptiXRenderResources,
  initialEye: Vector3,
  initialLookAt: Vector3,
  initialUp: Vector3,
  dispatcher: EventDispatcher,
  config: OrbitConfig = OrbitConfig()
) extends CameraHandler with SphericalOrbit with LazyLogging:

  override protected def orbitConfig: OrbitConfig = config

  // 4D rotation sensitivity - convert pixel deltas to rotation degrees
  private val rotation4DSensitivity = Const.Input.rotation4DSensitivity

  // Camera position state - LibGDX Vector3 is inherently mutable
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

  // Mouse tracking state - consolidated into Option[CameraDragState]
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var dragState: Option[CameraDragState] = None

  /** Check if shift key is currently pressed */
  private def isShiftPressed: Boolean =
    Gdx.input.isKeyPressed(Keys.SHIFT_LEFT) || Gdx.input.isKeyPressed(Keys.SHIFT_RIGHT)

  override protected def handleMouseDown(pos: ScreenCoords, button: MouseButton, pointer: Int): Boolean =
    dragState = Some(CameraDragState(pos, button))
    true

  override protected def handleMouseUp(pos: ScreenCoords, button: MouseButton, pointer: Int): Boolean =
    dragState = None
    true

  override protected def handleMouseDrag(pos: ScreenCoords, pointer: Int, button: MouseButton): Boolean =
    dragState match
      case None => false
      case Some(state) =>
        val deltaX = pos.x - state.lastPos.x
        val deltaY = pos.y - state.lastPos.y
        dragState = Some(state.copy(lastPos = pos))

        if isShiftPressed then
          handle4DRotation(button, deltaX, deltaY)
        else
          button match
            case MouseButton.Left => handleOrbit(deltaX, deltaY)
            case MouseButton.Right => handlePan(deltaX, deltaY)
            case _ => // Ignore other buttons

        true

  override protected def handleScroll(amountX: Float, amountY: Float): Boolean =
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

  private def handle4DRotation(button: MouseButton, deltaX: Int, deltaY: Int): Unit =
    button match
      case MouseButton.Left =>
        // Left drag: horizontal controls XW, vertical controls YW
        val rotXW = -deltaX * rotation4DSensitivity  // Negative for intuitive direction
        val rotYW = deltaY * rotation4DSensitivity   // Positive matches arrow key convention
        logger.debug(s"Shift + left drag: deltaX=$deltaX, deltaY=$deltaY -> rotXW=$rotXW, rotYW=$rotYW")
        dispatcher.notifyObservers(RotationProjectionParameters(rotXW, rotYW, 0f))
      case MouseButton.Right =>
        // Right drag: vertical controls ZW
        val rotZW = deltaY * rotation4DSensitivity
        logger.debug(s"Shift + right drag: deltaY=$deltaY -> rotZW=$rotZW")
        dispatcher.notifyObservers(RotationProjectionParameters(0f, 0f, rotZW))
      case _ =>
        // Ignore other buttons
