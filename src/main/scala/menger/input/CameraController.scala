package menger.input

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.Input.Buttons
import com.badlogic.gdx.Input.Keys
import com.badlogic.gdx.graphics.PerspectiveCamera
import com.badlogic.gdx.graphics.g3d.utils.CameraInputController
import menger.RotationProjectionParameters
import menger.Vec3

def isShiftPressed =
  Seq(Keys.SHIFT_LEFT, Keys.SHIFT_RIGHT).exists(Gdx.input.isKeyPressed)
def isLeftClicked = Gdx.input.isButtonPressed(Buttons.LEFT)
def isRightClicked = Gdx.input.isButtonPressed(Buttons.RIGHT)

class CameraController(
  camera: PerspectiveCamera, eventDispatcher: EventDispatcher
) extends CameraInputController(camera):

  // Touch input state tracking required by LibGDX CameraInputController framework
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var shiftStart = (x = 0, y = 0)

  override def touchDown(screenX: Int, screenY: Int, pointer: Int, button: Int): Boolean =
    shiftStart = (x = screenX, y = screenY)
    super.touchDown(screenX, screenY, pointer, button)

  override def touchDragged(screenX: Int, screenY: Int, pointer: Int): Boolean =
    if isShiftPressed then shiftTouchDragged(screenX, screenY)
    else super.touchDragged(screenX, screenY, pointer)

  override def scrolled(amountX: Float, amountY: Float): Boolean =
    if isShiftPressed then
      val eyeW = Math.pow(64, amountY.toDouble).toFloat + 1
      eventDispatcher.notifyObservers(RotationProjectionParameters(0,0,0, eyeW))
      false
    else super.scrolled(amountX, amountY)

  private def shiftTouchDragged(screenX: Int, screenY: Int): Boolean =
    val dragged = draggedDistance3D(screenX, screenY)
    shiftStart = (x = screenX, y = screenY)
    eventDispatcher.notifyObservers(RotationProjectionParameters.apply.tupled(dragged))
    false

  private def draggedDistance3D(screenX: Int, screenY: Int): Vec3[Float] =
    val screenDist = screenDistance(screenX, screenY)
    (screenToWorld(screenDist(0)), screenToWorld(screenDist(1)), screenToWorld(screenDist(2)))

  private def screenDistance(screenX: Int, screenY: Int): Vec3[Int] =
    if isLeftClicked then (x = screenX - shiftStart(0), y = shiftStart(1) - screenY, z = 0)
    else if isRightClicked then (x = 0, y = 0, z = screenX - shiftStart(0))
    else Vec3.zero

  private final val degrees = 360f
  private def screenToWorld(screen: Int): Float = screen.toFloat / Gdx.graphics.getWidth * degrees
