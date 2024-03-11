package menger.input

import com.badlogic.gdx.Input.Keys
import com.badlogic.gdx.InputAdapter

trait InputController  extends InputAdapter:
  protected var ctrl = false
  protected var alt = false
  protected var shift = false
  protected var rotatePressed: Map[Int, Boolean] = Map().withDefaultValue(false)

  def update(): Unit

  override def keyDown(keycode: Int): Boolean =
    keycode match
      case Keys.CONTROL_LEFT | Keys.CONTROL_RIGHT => setCtrl(true)
      case Keys.ALT_LEFT | Keys.ALT_RIGHT => setAlt(true)
      case Keys.SHIFT_LEFT | Keys.SHIFT_RIGHT => setShift(true)
      case Keys.LEFT | Keys.RIGHT | Keys.UP | Keys.DOWN => setRotatePressed(keycode, true)
      case _ => false

  override def keyUp(keycode: Int): Boolean =
    keycode match
      case Keys.CONTROL_LEFT | Keys.CONTROL_RIGHT => setCtrl(false)
      case Keys.ALT_LEFT | Keys.ALT_RIGHT => setAlt(false)
      case Keys.SHIFT_LEFT | Keys.SHIFT_RIGHT => setShift(false)
      case Keys.LEFT | Keys.RIGHT | Keys.UP | Keys.DOWN => setRotatePressed(keycode, false)
      case _ => false

  protected def setCtrl(mode: Boolean): Boolean =
    ctrl = mode
    true

  protected def setAlt(mode: Boolean): Boolean =
    alt = mode
    true

  protected def setShift(mode: Boolean): Boolean =
    shift = mode
    true

  protected def setRotatePressed(keycode: Int, mode: Boolean): Boolean =
    this.rotatePressed += keycode -> mode
    update()
    true
