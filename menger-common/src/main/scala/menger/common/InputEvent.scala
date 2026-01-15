package menger.common

/** Domain-specific keyboard key enumeration */
enum Key:
  case ControlLeft, ControlRight
  case AltLeft, AltRight
  case ShiftLeft, ShiftRight
  case Left, Right, Up, Down
  case PageUp, PageDown
  case Escape, Q
  case Unknown(code: Int)

/** Mouse button enumeration */
enum MouseButton:
  case Left, Right, Middle
  case Unknown(code: Int)

/** Screen coordinates for mouse events */
case class ScreenCoords(x: Int, y: Int)

/** Modifier key state (Ctrl, Alt, Shift) */
case class ModifierState(
  ctrl: Boolean = false,
  alt: Boolean = false,
  shift: Boolean = false
):
  def withCtrl(value: Boolean): ModifierState = copy(ctrl = value)
  def withAlt(value: Boolean): ModifierState = copy(alt = value)
  def withShift(value: Boolean): ModifierState = copy(shift = value)

/** Root sealed trait for all input events */
sealed trait InputEvent

object InputEvent:
  /** Base trait for keyboard events */
  sealed trait KeyEvent extends InputEvent:
    def key: Key
    def modifiers: ModifierState

  /** Key press event */
  case class KeyPress(key: Key, modifiers: ModifierState) extends KeyEvent

  /** Key release event */
  case class KeyRelease(key: Key, modifiers: ModifierState) extends KeyEvent

  /** Base trait for mouse events */
  sealed trait MouseEvent extends InputEvent:
    def position: ScreenCoords

  /** Mouse button down event */
  case class MouseDown(
    position: ScreenCoords,
    button: MouseButton,
    pointer: Int
  ) extends MouseEvent

  /** Mouse button up event */
  case class MouseUp(
    position: ScreenCoords,
    button: MouseButton,
    pointer: Int
  ) extends MouseEvent

  /** Mouse drag event */
  case class MouseDrag(
    position: ScreenCoords,
    pointer: Int,
    button: MouseButton
  ) extends MouseEvent

  /** Mouse scroll/wheel event */
  case class ScrollEvent(amountX: Float, amountY: Float) extends InputEvent
