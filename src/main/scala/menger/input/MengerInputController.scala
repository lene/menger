package menger.input

import com.badlogic.gdx.Input.Keys
import com.badlogic.gdx.graphics.PerspectiveCamera
import com.badlogic.gdx.math.Vector3
import com.badlogic.gdx.{Gdx, InputAdapter}

class MengerInputController(camera: PerspectiveCamera) extends InputAdapter:

  private var ctrl = false
  private var alt = false
  private var shift = false
  private var rotatePressed: Map[Int, Boolean] = Map().withDefaultValue(false)

  private var angleXW = 0f
  private var angleYW = 0f
  private var angleZW = 0f

  private val defaultPos = camera.position.cpy
  private val defaultDirection = camera.direction.cpy
  private val defaultUp = camera.up.cpy

  override def keyDown(keycode: Int): Boolean =
    keycode match
      case Keys.CONTROL_LEFT | Keys.CONTROL_RIGHT => setCtrl(true)
      case Keys.ALT_LEFT | Keys.ALT_RIGHT => setAlt(true)
      case Keys.SHIFT_LEFT | Keys.SHIFT_RIGHT => setShift(true)
      case Keys.LEFT | Keys.RIGHT | Keys.UP | Keys.DOWN => setRotatePressed(keycode, true)
      case Keys.PAGE_DOWN | Keys.PAGE_UP => setRotatePressed(keycode, true)
      case Keys.ESCAPE => resetCamera
      case Keys.Q =>
        if ctrl then System.exit(0)
        true
      case _ => false

  override def keyUp(keycode: Int): Boolean =
    keycode match
      case Keys.CONTROL_LEFT | Keys.CONTROL_RIGHT => setCtrl(false)
      case Keys.ALT_LEFT | Keys.ALT_RIGHT => setAlt(false)
      case Keys.SHIFT_LEFT | Keys.SHIFT_RIGHT => setShift(false)
      case Keys.LEFT | Keys.RIGHT | Keys.UP | Keys.DOWN => setRotatePressed(keycode, false)
      case Keys.PAGE_DOWN | Keys.PAGE_UP => setRotatePressed(keycode, false)
      case _ => false

  private final val rotateAngle = 360f
  private final val origin = Vector3.Zero
  def update(): Unit =
    Gdx.app.log("InputControllerHigherD", s"rotatePressed:$rotatePressed, angleXW: $angleXW, angleYW: $angleYW, angleZW: $angleZW, $shift, $ctrl, $alt")
    if rotatePressed.values.exists(_ == true) then
      val delta = Gdx.graphics.getDeltaTime
      if !(shift || ctrl || alt) then
        // algorithm pulled from
        // https://github.com/libgdx/libgdx/blob/master/gdx/src/com/badlogic/gdx/graphics/g3d/utils/CameraInputController.java#L187
        if (rotatePressed(Keys.RIGHT)) camera.rotateAround(origin, Vector3.Y, -delta*rotateAngle)
        if (rotatePressed(Keys.LEFT)) camera.rotateAround(origin, Vector3.Y, delta*rotateAngle)
        val tmpXZ = Vector3()
        tmpXZ.set(camera.direction).crs(camera.up).y = 0f
        if (rotatePressed(Keys.UP)) camera.rotateAround(origin, tmpXZ.nor, delta*rotateAngle)
        if (rotatePressed(Keys.DOWN)) camera.rotateAround(origin, tmpXZ.nor, -delta*rotateAngle)
        camera.update()
      else if shift then
        if rotatePressed(Keys.RIGHT) then angleXW -= delta * rotateAngle
        if rotatePressed(Keys.LEFT) then angleXW += delta * rotateAngle
        if rotatePressed(Keys.UP) then angleYW += delta * rotateAngle
        if rotatePressed(Keys.DOWN) then angleYW -= delta * rotateAngle
        if rotatePressed(Keys.PAGE_UP) then angleZW += delta * rotateAngle
        if rotatePressed(Keys.PAGE_DOWN) then angleZW -= delta * rotateAngle

  private def resetCamera: Boolean =
    camera.position.set(defaultPos)
    camera.direction.set(defaultDirection)
    camera.up.set(defaultUp)
    camera.lookAt(0, 0, 0)
    camera.update()
    true

  private def setCtrl(mode: Boolean): Boolean =
    ctrl = mode
    true

  private def setAlt(mode: Boolean): Boolean =
    alt = mode
    true

  private def setShift(mode: Boolean): Boolean =
    shift = mode
    true

  private def setRotatePressed(keycode: Int, mode: Boolean): Boolean =
    this.rotatePressed += keycode -> mode
    update()
    true
