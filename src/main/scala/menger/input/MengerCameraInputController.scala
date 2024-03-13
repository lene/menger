package menger.input

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.PerspectiveCamera
import com.badlogic.gdx.graphics.g3d.utils.CameraInputController
import com.badlogic.gdx.Input.Keys

class MengerCameraInputController(camera: PerspectiveCamera) extends CameraInputController(camera):

  private var thisStartX = 0
  private var thisStartY = 0

  override def touchDown(screenX: Int, screenY: Int, pointer: Int, button: Int): Boolean =
    thisStartX = screenX
    thisStartY = screenY
    super.touchDown(screenX, screenY, pointer, button)

  override def touchDragged(screenX: Int, screenY: Int, pointer: Int): Boolean =
    if Gdx.input.isKeyPressed(Keys.SHIFT_LEFT) || Gdx.input.isKeyPressed(Keys.SHIFT_RIGHT) then
      val deltaX = (screenX - thisStartX).toFloat / Gdx.graphics.getWidth
      val deltaY = (thisStartY - screenY).toFloat / Gdx.graphics.getHeight
      thisStartX = screenX
      thisStartY = screenY
      Gdx.app.log("MengerCameraInputController", s"touchDragged($screenX, $screenY, $pointer) -> $deltaX, $deltaY")
      true
    else super.touchDragged(screenX, screenY, pointer)
