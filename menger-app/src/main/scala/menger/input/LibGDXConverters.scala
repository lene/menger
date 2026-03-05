package menger.input

import com.badlogic.gdx.Input.Buttons
import com.badlogic.gdx.Input.Keys
import menger.common.Key
import menger.common.MouseButton

/** Converters from LibGDX key/button codes to domain types */
object LibGDXConverters:

  /** Convert LibGDX key code to domain Key enum */
  def convertKey(gdxKeyCode: Int): Key = gdxKeyCode match
    case Keys.CONTROL_LEFT => Key.ControlLeft
    case Keys.CONTROL_RIGHT => Key.ControlRight
    case Keys.ALT_LEFT => Key.AltLeft
    case Keys.ALT_RIGHT => Key.AltRight
    case Keys.SHIFT_LEFT => Key.ShiftLeft
    case Keys.SHIFT_RIGHT => Key.ShiftRight
    case Keys.LEFT => Key.Left
    case Keys.RIGHT => Key.Right
    case Keys.UP => Key.Up
    case Keys.DOWN => Key.Down
    case Keys.PAGE_UP => Key.PageUp
    case Keys.PAGE_DOWN => Key.PageDown
    case Keys.ESCAPE => Key.Escape
    case Keys.Q => Key.Q
    case _ => Key.Unknown(gdxKeyCode)

  /** Convert LibGDX button code to domain MouseButton enum */
  def convertButton(gdxButton: Int): MouseButton = gdxButton match
    case Buttons.LEFT => MouseButton.Left
    case Buttons.RIGHT => MouseButton.Right
    case Buttons.MIDDLE => MouseButton.Middle
    case _ => MouseButton.Unknown(gdxButton)

  /** Convert domain MouseButton to LibGDX button code */
  def toGdxButton(button: MouseButton): Int = button match
    case MouseButton.Left        => Buttons.LEFT
    case MouseButton.Right       => Buttons.RIGHT
    case MouseButton.Middle      => Buttons.MIDDLE
    case MouseButton.Unknown(c)  => c
