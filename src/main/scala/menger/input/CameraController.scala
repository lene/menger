package menger.input

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.PerspectiveCamera
import com.badlogic.gdx.graphics.g3d.utils.CameraInputController
import com.badlogic.gdx.Input.{Buttons, Keys}
import menger.RotationProjectionParameters

def isShiftPressed =
  Seq(Keys.SHIFT_LEFT, Keys.SHIFT_RIGHT).exists(Gdx.input.isKeyPressed)
def isLeftClicked = Gdx.input.isButtonPressed(Buttons.LEFT)
def isRightClicked = Gdx.input.isButtonPressed(Buttons.RIGHT)

class CameraController(
  camera: PerspectiveCamera, eventDispatcher: EventDispatcher
) extends CameraInputController(camera):

  private var shiftStart = (0, 0)

  override def touchDown(screenX: Int, screenY: Int, pointer: Int, button: Int): Boolean =
    shiftStart = (screenX, screenY)
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
    shiftStart = (screenX, screenY)
    eventDispatcher.notifyObservers(RotationProjectionParameters.apply.tupled(dragged))
    false

  private def draggedDistance3D(screenX: Int, screenY: Int): (Float, Float, Float) =
    val screenDist = screenDistance(screenX, screenY)
    (screenToWorld(screenDist(0)), screenToWorld(screenDist(1)), screenToWorld(screenDist(2)))

  private def screenDistance(screenX: Int, screenY: Int) =
    if isLeftClicked then (screenX - shiftStart(0), shiftStart(1) - screenY, 0)
    else if isRightClicked then (0, 0, screenX - shiftStart(0))
    else (0, 0, 0)

  private final val degrees = 360f
  private def screenToWorld(screen: Int): Float = screen.toFloat / Gdx.graphics.getWidth * degrees
