package menger.input

import com.badlogic.gdx.Input.Buttons
import com.badlogic.gdx.Input.Keys
import com.badlogic.gdx.graphics.PerspectiveCamera
import com.badlogic.gdx.graphics.g3d.utils.CameraInputController
import menger.RotationProjectionParameters
import menger.common.Const
import menger.common.MouseButton
import menger.common.ScreenCoords
import menger.gdx.DragTracker
import menger.gdx.GdxRuntime

/**
 * Camera/mouse input handler for LibGDX 3D rendering mode.
 *
 * Uses composition over inheritance - delegates standard camera operations
 * (zoom, pan, rotate) to LibGDX's CameraInputController, and intercepts
 * shift-modified operations for 4D rotation.
 *
 * Handles:
 * - Left drag: 3D camera orbit (without shift) or 4D XW/YW rotation (with shift)
 * - Right drag: 3D camera pan (without shift) or 4D ZW rotation (with shift)
 * - Scroll: 3D camera zoom (without shift) or 4D eyeW adjustment (with shift)
 */
class GdxCameraHandler(
  camera: PerspectiveCamera,
  eventDispatcher: EventDispatcher
) extends CameraHandler:

  /** Delegate standard camera operations to LibGDX controller */
  private val baseController = CameraInputController(camera)

  /** Track drag start position for 4D rotation calculations */
  private val dragTracker = DragTracker()

  /** Check if shift key is currently pressed */
  private def isShiftPressed: Boolean =
    GdxRuntime.isKeyPressed(Keys.SHIFT_LEFT) || GdxRuntime.isKeyPressed(Keys.SHIFT_RIGHT)

  /** Check if left mouse button is pressed */
  private def isLeftClicked: Boolean =
    GdxRuntime.isButtonPressed(Buttons.LEFT)

  /** Check if right mouse button is pressed */
  private def isRightClicked: Boolean =
    GdxRuntime.isButtonPressed(Buttons.RIGHT)

  override protected def handleMouseDown(pos: ScreenCoords, button: MouseButton, pointer: Int): Boolean =
    dragTracker.start(pos)
    baseController.touchDown(pos.x, pos.y, pointer, LibGDXConverters.toGdxButton(button))

  override protected def handleMouseUp(pos: ScreenCoords, button: MouseButton, pointer: Int): Boolean =
    baseController.touchUp(pos.x, pos.y, pointer, LibGDXConverters.toGdxButton(button))

  override protected def handleMouseDrag(pos: ScreenCoords, pointer: Int, button: MouseButton): Boolean =
    if isShiftPressed then
      shiftTouchDragged(pos)
    else
      baseController.touchDragged(pos.x, pos.y, pointer)

  override protected def handleScroll(amountX: Float, amountY: Float): Boolean =
    if isShiftPressed then
      val eyeW = Math.pow(Const.Input.eyeScrollBase, amountY.toDouble).toFloat + Const.Input.eyeScrollOffset
      eventDispatcher.notifyObservers(RotationProjectionParameters(0, 0, 0, eyeW))
      false
    else
      baseController.scrolled(amountX, amountY)

  private def shiftTouchDragged(pos: ScreenCoords): Boolean =
    val (rotXW, rotYW, rotZW) = draggedDistance3D(pos)
    dragTracker.start(pos)
    eventDispatcher.notifyObservers(RotationProjectionParameters(rotXW, rotYW, rotZW))
    false

  private def draggedDistance3D(pos: ScreenCoords): (Float, Float, Float) =
    val (screenX, screenY, screenZ) = screenDistance(pos)
    (screenToWorld(screenX), screenToWorld(screenY), screenToWorld(screenZ))

  private def screenDistance(pos: ScreenCoords): (Int, Int, Int) =
    if isLeftClicked then
      (pos.x - dragTracker.origin.x, dragTracker.origin.y - pos.y, 0)
    else if isRightClicked then
      (0, 0, pos.x - dragTracker.origin.x)
    else
      (0, 0, 0)

  private val degrees = Const.Input.fullRotationDegrees
  private def screenToWorld(screen: Int): Float =
    val w = GdxRuntime.width
    if w > 0 then screen.toFloat / w * degrees else 0f

