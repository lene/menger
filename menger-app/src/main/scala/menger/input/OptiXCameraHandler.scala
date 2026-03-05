package menger.input

import com.badlogic.gdx.Input.Keys
import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.OptiXRenderResources
import menger.RotationProjectionParameters
import menger.common.Const
import menger.common.MouseButton
import menger.common.ScreenCoords
import menger.gdx.GdxRuntime
import menger.gdx.OrbitCamera
import menger.optix.CameraState
import menger.optix.OptiXRendererWrapper

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
 * Delegates all mutable camera state to OrbitCamera.
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
) extends CameraHandler with LazyLogging:

  // All mutable camera state lives in OrbitCamera
  private val camera = OrbitCamera(initialEye, initialLookAt, initialUp, config)

  // 4D rotation sensitivity - convert pixel deltas to rotation degrees
  private val rotation4DSensitivity = Const.Input.rotation4DSensitivity

  // Public getters for camera state (needed for scene rebuild)
  def currentEye: Vector3    = camera.currentEye
  def currentLookAt: Vector3 = camera.currentLookAt
  def currentUp: Vector3     = camera.currentUp

  private def isShiftPressed: Boolean =
    GdxRuntime.isKeyPressed(Keys.SHIFT_LEFT) || GdxRuntime.isKeyPressed(Keys.SHIFT_RIGHT)

  override protected def handleMouseDown(pos: ScreenCoords, button: MouseButton, pointer: Int): Boolean =
    camera.startDrag(pos, button)
    true

  override protected def handleMouseUp(pos: ScreenCoords, button: MouseButton, pointer: Int): Boolean =
    camera.endDrag()
    true

  override protected def handleMouseDrag(pos: ScreenCoords, pointer: Int, button: MouseButton): Boolean =
    camera.moveDrag(pos) match
      case None => false
      case Some((deltaX, deltaY, btn)) =>
        if isShiftPressed then
          handle4DRotation(btn, deltaX, deltaY)
        else
          btn match
            case MouseButton.Left  => handleOrbit(deltaX, deltaY)
            case MouseButton.Right => handlePan(deltaX, deltaY)
            case _                 => // Ignore other buttons
        true

  override protected def handleScroll(amountX: Float, amountY: Float): Boolean =
    if isShiftPressed then
      val eyeW = computeEyeW(amountY)
      dispatcher.notifyObservers(RotationProjectionParameters(0, 0, 0, eyeW))
    else
      handleZoom(amountY)
    true

  private def handleOrbit(deltaX: Int, deltaY: Int): Unit =
    camera.orbit(deltaX, deltaY)
    updateCamera()

  private def handlePan(deltaX: Int, deltaY: Int): Unit =
    camera.pan(deltaX, deltaY)
    updateCamera()

  private def handleZoom(scrollAmount: Float): Unit =
    camera.zoom(scrollAmount)
    updateCamera()

  private def updateCamera(): Unit =
    cameraState.updateCamera(rendererWrapper.renderer, camera.currentEye, camera.currentLookAt, camera.currentUp)
    renderResources.markNeedsRender()
    GdxRuntime.requestRendering()
    logger.debug(s"Camera updated: eye=${camera.currentEye}, lookAt=${camera.currentLookAt}")

  private def handle4DRotation(button: MouseButton, deltaX: Int, deltaY: Int): Unit =
    button match
      case MouseButton.Left =>
        val rotXW = -deltaX * rotation4DSensitivity
        val rotYW = deltaY * rotation4DSensitivity
        logger.debug(s"Shift + left drag: deltaX=$deltaX, deltaY=$deltaY -> rotXW=$rotXW, rotYW=$rotYW")
        dispatcher.notifyObservers(RotationProjectionParameters(rotXW, rotYW, 0f))
      case MouseButton.Right =>
        val rotZW = deltaY * rotation4DSensitivity
        logger.debug(s"Shift + right drag: deltaY=$deltaY -> rotZW=$rotZW")
        dispatcher.notifyObservers(RotationProjectionParameters(0f, 0f, rotZW))
      case _ =>
        // Ignore other buttons
