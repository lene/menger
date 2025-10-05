package menger.input

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.Input.Keys
import com.badlogic.gdx.InputAdapter
import com.badlogic.gdx.graphics.PerspectiveCamera
import com.badlogic.gdx.math.Vector3
import menger.RotationProjectionParameters

class KeyController(camera: PerspectiveCamera, dispatcher: EventDispatcher) extends InputAdapter:

  // Input state tracking required by LibGDX InputAdapter framework
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var ctrlPressed = false
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var altPressed = false
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var shiftPressed = false
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var rotatePressed: Map[Int, Boolean] = Map().withDefaultValue(false)

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
        false
      case _ => false

  override def keyUp(keycode: Int): Boolean =
    keycode match
      case Keys.CONTROL_LEFT | Keys.CONTROL_RIGHT => setCtrl(false)
      case Keys.ALT_LEFT | Keys.ALT_RIGHT => setAlt(false)
      case Keys.SHIFT_LEFT | Keys.SHIFT_RIGHT => setShift(false)
      case Keys.LEFT | Keys.RIGHT | Keys.UP | Keys.DOWN => setRotatePressed(keycode, false)
      case Keys.PAGE_DOWN | Keys.PAGE_UP => setRotatePressed(keycode, false)
      case _ => false

  private final val rotateAngle = 45f
  def update(): Unit =
    if rotatePressed.values.exists(_ == true) then
      val delta = Gdx.graphics.getDeltaTime
      if !(shift || ctrl || alt) then onNoModifiersPressed(delta)
      else if shift then onShiftPressed(delta)

  def shift: Boolean = shiftPressed
  def ctrl: Boolean = ctrlPressed
  def alt: Boolean = altPressed

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

  private def resetCamera: Boolean =
    camera.position.set(defaultPos)
    camera.direction.set(defaultDirection)
    camera.up.set(defaultUp)
    camera.lookAt(0, 0, 0)
    camera.update()
    false

  private def setCtrl(mode: Boolean): Boolean =
    ctrlPressed = mode
    false

  private def setAlt(mode: Boolean): Boolean =
    altPressed = mode
    false

  private def setShift(mode: Boolean): Boolean =
    shiftPressed = mode
    false

  private def setRotatePressed(keycode: Int, mode: Boolean): Boolean =
    this.rotatePressed += keycode -> mode
    update()
    false
