package menger.input

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.Input.Keys
import com.badlogic.gdx.graphics.PerspectiveCamera
import com.badlogic.gdx.math.Vector3
import menger.RotationProjectionParameters
import menger.common.Const

class GdxKeyController(camera: PerspectiveCamera, dispatcher: EventDispatcher) extends BaseKeyController:

  private val defaultPos = camera.position.cpy
  private val defaultDirection = camera.direction.cpy
  private val defaultUp = camera.up.cpy

  private final val rotateAngle = Const.Input.defaultRotateAngle

  def update(): Unit =
    if rotatePressed.values.exists(_ == true) then
      val delta = Gdx.graphics.getDeltaTime
      if !(shift || ctrl || alt) then onNoModifiersPressed(delta)
      else if shift then onShiftPressed(delta)

  override protected def handleEscape(): Boolean =
    resetCamera()
    false

  override protected def onRotationUpdate(): Unit = update()

  private final val origin = Vector3.Zero
  private def onNoModifiersPressed(delta: Float): Unit =
    // algorithm pulled from
    // https://github.com/libgdx/libgdx/blob/master/gdx/src/com/badlogic/gdx/graphics/g3d/utils/CameraInputController.java#L187
    camera.rotateAround(origin, Vector3.Y, angle(delta, Seq(Keys.RIGHT, Keys.LEFT)))
    val tmpXZ = Vector3()
    tmpXZ.set(camera.direction).crs(camera.up).y = 0f
    camera.rotateAround(origin, tmpXZ.nor, angle(delta, Seq(Keys.UP, Keys.DOWN)))
    camera.update()

  private def onShiftPressed(delta: Float): Unit =
    dispatcher.notifyObservers(
      RotationProjectionParameters(
        angle(delta, Seq(Keys.LEFT, Keys.RIGHT)), angle(delta, Seq(Keys.UP, Keys.DOWN)),
        angle(delta, Seq(Keys.PAGE_UP, Keys.PAGE_DOWN))
      )
    )

  private val factor = Map(
    Keys.RIGHT -> -1, Keys.LEFT -> 1, Keys.UP -> 1, Keys.DOWN -> -1,
    Keys.PAGE_UP -> 1, Keys.PAGE_DOWN -> -1
  )
  private def angle(delta: Float, keys: Seq[Int]): Float = delta * rotateAngle * direction(keys)
  private def direction(keys: Seq[Int]) = keys.find(rotatePressed).map(factor(_)).getOrElse(0)

  private def resetCamera(): Unit =
    camera.position.set(defaultPos)
    camera.direction.set(defaultDirection)
    camera.up.set(defaultUp)
    camera.lookAt(0, 0, 0)
    camera.update()
