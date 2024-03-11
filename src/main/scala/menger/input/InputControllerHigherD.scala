package menger.input

import com.badlogic.gdx.Input.Keys
import com.badlogic.gdx.graphics.PerspectiveCamera
import com.badlogic.gdx.math.Vector3
import com.badlogic.gdx.{Gdx, InputAdapter}

class InputControllerHigherD extends InputController:
  private var angleXW = 0f
  private var angleYW = 0f
  private var angleZW = 0f

  override def keyDown(keycode: Int): Boolean =
    val caughtByParent = super.keyDown(keycode)
    if caughtByParent then true
    else keycode match
      case Keys.PAGE_DOWN | Keys.PAGE_UP => setRotatePressed(keycode, true)
      case _ => false
      
  override def keyUp(keycode: Int): Boolean =
    val caughtByParent = super.keyUp(keycode)
    if caughtByParent then true
    else keycode match
      case Keys.PAGE_DOWN | Keys.PAGE_UP => setRotatePressed(keycode, false)
      case _ => false
      
  // algorithm pulled from
  // https://github.com/libgdx/libgdx/blob/master/gdx/src/com/badlogic/gdx/graphics/g3d/utils/CameraInputController.java#L187
  private final val rotateAngle = 360f
  def update(): Unit =
    if rotatePressed.values.exists(_ == true) && shift then
      val delta = Gdx.graphics.getDeltaTime
      if rotatePressed(Keys.RIGHT) then angleXW -= delta*rotateAngle
      if rotatePressed(Keys.LEFT) then angleXW += delta*rotateAngle
      if rotatePressed(Keys.UP) then angleYW += delta*rotateAngle
      if rotatePressed(Keys.DOWN) then angleYW -= delta*rotateAngle
      if rotatePressed(Keys.PAGE_UP) then angleZW += delta*rotateAngle
      if rotatePressed(Keys.PAGE_DOWN) then angleZW -= delta*rotateAngle
//      camera.update()
