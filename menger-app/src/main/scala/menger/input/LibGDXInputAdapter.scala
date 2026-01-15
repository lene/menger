package menger.input

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.Input.Buttons
import com.badlogic.gdx.Input.Keys
import com.badlogic.gdx.InputAdapter
import menger.common.InputEvent
import menger.common.Key
import menger.common.ModifierState
import menger.common.MouseButton
import menger.common.ScreenCoords

/**
 * Adapter that bridges LibGDX input events to domain input events.
 *
 * This class isolates all LibGDX coupling to a single layer, converting
 * LibGDX callback methods into domain-level InputEvent instances that
 * are dispatched to handlers.
 *
 * @param handlers Sequence of input handlers to receive domain events
 */
class LibGDXInputAdapter(handlers: Seq[InputHandler]) extends InputAdapter:

  /**
   * Query current modifier state from Gdx.input.
   * This ensures modifier state is always accurate even if we miss events.
   */
  private def currentModifiers: ModifierState = ModifierState(
    ctrl = isKeyPressed(Key.ControlLeft) || isKeyPressed(Key.ControlRight),
    alt = isKeyPressed(Key.AltLeft) || isKeyPressed(Key.AltRight),
    shift = isKeyPressed(Key.ShiftLeft) || isKeyPressed(Key.ShiftRight)
  )

  /** Check if a domain Key is currently pressed via Gdx.input */
  private def isKeyPressed(key: Key): Boolean = key match
    case Key.ControlLeft => Gdx.input.isKeyPressed(Keys.CONTROL_LEFT)
    case Key.ControlRight => Gdx.input.isKeyPressed(Keys.CONTROL_RIGHT)
    case Key.ShiftLeft => Gdx.input.isKeyPressed(Keys.SHIFT_LEFT)
    case Key.ShiftRight => Gdx.input.isKeyPressed(Keys.SHIFT_RIGHT)
    case Key.AltLeft => Gdx.input.isKeyPressed(Keys.ALT_LEFT)
    case Key.AltRight => Gdx.input.isKeyPressed(Keys.ALT_RIGHT)
    case _ => false

  override def keyDown(keycode: Int): Boolean =
    val key = LibGDXConverters.convertKey(keycode)
    val event = InputEvent.KeyPress(key, currentModifiers)
    handlers.exists(_.handleInput(event))

  override def keyUp(keycode: Int): Boolean =
    val key = LibGDXConverters.convertKey(keycode)
    val event = InputEvent.KeyRelease(key, currentModifiers)
    handlers.exists(_.handleInput(event))

  override def touchDown(screenX: Int, screenY: Int, pointer: Int, button: Int): Boolean =
    val event = InputEvent.MouseDown(
      ScreenCoords(screenX, screenY),
      LibGDXConverters.convertButton(button),
      pointer
    )
    handlers.exists(_.handleInput(event))

  override def touchUp(screenX: Int, screenY: Int, pointer: Int, button: Int): Boolean =
    val event = InputEvent.MouseUp(
      ScreenCoords(screenX, screenY),
      LibGDXConverters.convertButton(button),
      pointer
    )
    handlers.exists(_.handleInput(event))

  override def touchDragged(screenX: Int, screenY: Int, pointer: Int): Boolean =
    // Determine which button is pressed for the drag operation
    val button =
      if Gdx.input.isButtonPressed(Buttons.LEFT) then MouseButton.Left
      else if Gdx.input.isButtonPressed(Buttons.RIGHT) then MouseButton.Right
      else if Gdx.input.isButtonPressed(Buttons.MIDDLE) then MouseButton.Middle
      else MouseButton.Unknown(-1)

    val event = InputEvent.MouseDrag(ScreenCoords(screenX, screenY), pointer, button)
    handlers.exists(_.handleInput(event))

  override def scrolled(amountX: Float, amountY: Float): Boolean =
    val event = InputEvent.ScrollEvent(amountX, amountY)
    handlers.exists(_.handleInput(event))
