package menger.input

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.Input.Keys
import com.badlogic.gdx.InputAdapter

abstract class BaseKeyController extends InputAdapter:

  // Input state tracking required by LibGDX InputAdapter framework
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  protected var ctrlPressed = false
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  protected var altPressed = false
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  protected var shiftPressed = false
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  protected var rotatePressed: Map[Int, Boolean] = Map().withDefaultValue(false)

  def shift: Boolean = shiftPressed
  def ctrl: Boolean = ctrlPressed
  def alt: Boolean = altPressed

  override def keyDown(keycode: Int): Boolean =
    keycode match
      case Keys.CONTROL_LEFT | Keys.CONTROL_RIGHT => setCtrl(true)
      case Keys.ALT_LEFT | Keys.ALT_RIGHT => setAlt(true)
      case Keys.SHIFT_LEFT | Keys.SHIFT_RIGHT => setShift(true)
      case Keys.LEFT | Keys.RIGHT | Keys.UP | Keys.DOWN => setRotatePressed(keycode, true)
      case Keys.PAGE_DOWN | Keys.PAGE_UP => setRotatePressed(keycode, true)
      case Keys.ESCAPE => handleEscape()
      case Keys.Q if ctrl => handleQuit()
      case _ => false

  override def keyUp(keycode: Int): Boolean =
    keycode match
      case Keys.CONTROL_LEFT | Keys.CONTROL_RIGHT => setCtrl(false)
      case Keys.ALT_LEFT | Keys.ALT_RIGHT => setAlt(false)
      case Keys.SHIFT_LEFT | Keys.SHIFT_RIGHT => setShift(false)
      case Keys.LEFT | Keys.RIGHT | Keys.UP | Keys.DOWN => setRotatePressed(keycode, false)
      case Keys.PAGE_DOWN | Keys.PAGE_UP => setRotatePressed(keycode, false)
      case _ => false

  protected def handleQuit(): Boolean =
    Gdx.app.exit()
    true

  protected def handleEscape(): Boolean

  protected def onRotationUpdate(): Unit

  private def setCtrl(mode: Boolean): Boolean =
    ctrlPressed = mode
    false

  private def setAlt(mode: Boolean): Boolean =
    altPressed = mode
    false

  private def setShift(mode: Boolean): Boolean =
    shiftPressed = mode
    false

  protected def setRotatePressed(keycode: Int, mode: Boolean): Boolean =
    rotatePressed = rotatePressed.updated(keycode, mode)
    onRotationUpdate()
    false
