package menger.input

import menger.common.InputEvent
import menger.common.Key
import menger.common.ModifierState
import menger.common.MouseButton
import menger.common.ScreenCoords

/** Base trait for all input handlers */
trait InputHandler:
  def handleInput(event: InputEvent): Boolean

/** Specialized handler for keyboard input */
trait KeyHandler extends InputHandler:
  /** Track current modifier state */
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  protected var modifierState: ModifierState = ModifierState()

  def handleInput(event: InputEvent): Boolean = event match
    case InputEvent.KeyPress(key, modifiers) =>
      updateModifierState(key, pressed = true)
      handleKeyPress(key, modifiers)
    case InputEvent.KeyRelease(key, modifiers) =>
      updateModifierState(key, pressed = false)
      handleKeyRelease(key, modifiers)
    case _ => false

  protected def handleKeyPress(key: Key, modifiers: ModifierState): Boolean
  protected def handleKeyRelease(key: Key, modifiers: ModifierState): Boolean

  /** Update internal modifier state tracking */
  private def updateModifierState(key: Key, pressed: Boolean): Unit =
    key match
      case Key.ControlLeft | Key.ControlRight =>
        modifierState = modifierState.withCtrl(pressed)
      case Key.AltLeft | Key.AltRight =>
        modifierState = modifierState.withAlt(pressed)
      case Key.ShiftLeft | Key.ShiftRight =>
        modifierState = modifierState.withShift(pressed)
      case _ => // Other keys don't affect modifier state

/** Specialized handler for camera/mouse input */
trait CameraHandler extends InputHandler:
  def handleInput(event: InputEvent): Boolean = event match
    case InputEvent.MouseDown(pos, button, pointer) =>
      handleMouseDown(pos, button, pointer)
    case InputEvent.MouseUp(pos, button, pointer) =>
      handleMouseUp(pos, button, pointer)
    case InputEvent.MouseDrag(pos, pointer, button) =>
      handleMouseDrag(pos, pointer, button)
    case InputEvent.ScrollEvent(amountX, amountY) =>
      handleScroll(amountX, amountY)
    case _ => false

  protected def handleMouseDown(pos: ScreenCoords, button: MouseButton, pointer: Int): Boolean
  protected def handleMouseUp(pos: ScreenCoords, button: MouseButton, pointer: Int): Boolean
  protected def handleMouseDrag(pos: ScreenCoords, pointer: Int, button: MouseButton): Boolean
  protected def handleScroll(amountX: Float, amountY: Float): Boolean
