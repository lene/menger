package menger.input

import com.badlogic.gdx.Input.Keys
import com.badlogic.gdx.graphics.PerspectiveCamera
import com.badlogic.gdx.math.Vector3
import com.badlogic.gdx.{Gdx, InputAdapter}

class InputController3D(camera: PerspectiveCamera) extends InputController:

  private val defaultPos = camera.position.cpy
  private val defaultDirection = camera.direction.cpy
  private val defaultUp = camera.up.cpy
  
  override def keyDown(keycode: Int): Boolean =
    val caughtByParent = super.keyDown(keycode)
    if caughtByParent then true
    else keycode match
      case Keys.ESCAPE => resetCamera
      case Keys.Q =>
        if ctrl then System.exit(0)
        true
      case _ => false

  // algorithm pulled from
  // https://github.com/libgdx/libgdx/blob/master/gdx/src/com/badlogic/gdx/graphics/g3d/utils/CameraInputController.java#L187
  private final val rotateAngle = 360f
  private final val origin = Vector3.Zero
  def update(): Unit =
    if rotatePressed.values.exists(_ == true) && !(shift || ctrl || alt) then
      val delta = Gdx.graphics.getDeltaTime
      if (rotatePressed(Keys.RIGHT)) camera.rotateAround(origin, Vector3.Y, -delta*rotateAngle)
      if (rotatePressed(Keys.LEFT)) camera.rotateAround(origin, Vector3.Y, delta*rotateAngle)
      val tmpXZ = Vector3()
      tmpXZ.set(camera.direction).crs(camera.up).y = 0f
      if (rotatePressed(Keys.UP)) camera.rotateAround(origin, tmpXZ.nor, delta*rotateAngle)
      if (rotatePressed(Keys.DOWN)) camera.rotateAround(origin, tmpXZ.nor, -delta*rotateAngle)
      camera.update()

  private def resetCamera: Boolean =
    camera.position.set(defaultPos)
    camera.direction.set(defaultDirection)
    camera.up.set(defaultUp)
    camera.lookAt(0, 0, 0)
    camera.update()
    true
