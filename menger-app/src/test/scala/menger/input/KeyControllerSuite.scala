package menger.input

import com.badlogic.gdx.Input.Keys
import menger.common.InputEvent
import menger.common.Key
import menger.common.ModifierState
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class KeyControllerSuite extends AnyFlatSpec with Matchers:

  // Helper to create a dispatcher (can be a no-op for most tests)
  private def createDispatcher(): EventDispatcher = EventDispatcher()
  private val emptyModifiers = ModifierState()

  "KeyHandler modifier tracking" should "track Ctrl key press" in:
    val handler = OptiXKeyHandler(createDispatcher())
    handler.currentModifiers.ctrl shouldBe false

    handler.handleInput(InputEvent.KeyPress(Key.ControlLeft, emptyModifiers))
    handler.currentModifiers.ctrl shouldBe true

    handler.handleInput(InputEvent.KeyRelease(Key.ControlLeft, emptyModifiers))
    handler.currentModifiers.ctrl shouldBe false

  it should "track Ctrl right key press" in:
    val handler = OptiXKeyHandler(createDispatcher())
    handler.handleInput(InputEvent.KeyPress(Key.ControlRight, emptyModifiers))
    handler.currentModifiers.ctrl shouldBe true

    handler.handleInput(InputEvent.KeyRelease(Key.ControlRight, emptyModifiers))
    handler.currentModifiers.ctrl shouldBe false

  it should "track Shift key press" in:
    val handler = OptiXKeyHandler(createDispatcher())
    handler.currentModifiers.shift shouldBe false

    handler.handleInput(InputEvent.KeyPress(Key.ShiftLeft, emptyModifiers))
    handler.currentModifiers.shift shouldBe true

    handler.handleInput(InputEvent.KeyRelease(Key.ShiftLeft, emptyModifiers))
    handler.currentModifiers.shift shouldBe false

  it should "track Shift right key press" in:
    val handler = OptiXKeyHandler(createDispatcher())
    handler.handleInput(InputEvent.KeyPress(Key.ShiftRight, emptyModifiers))
    handler.currentModifiers.shift shouldBe true

  it should "track Alt key press" in:
    val handler = OptiXKeyHandler(createDispatcher())
    handler.currentModifiers.alt shouldBe false

    handler.handleInput(InputEvent.KeyPress(Key.AltLeft, emptyModifiers))
    handler.currentModifiers.alt shouldBe true

    handler.handleInput(InputEvent.KeyRelease(Key.AltLeft, emptyModifiers))
    handler.currentModifiers.alt shouldBe false

  it should "track Alt right key press" in:
    val handler = OptiXKeyHandler(createDispatcher())
    handler.handleInput(InputEvent.KeyPress(Key.AltRight, emptyModifiers))
    handler.currentModifiers.alt shouldBe true

  it should "track multiple modifiers simultaneously" in:
    val handler = OptiXKeyHandler(createDispatcher())

    handler.handleInput(InputEvent.KeyPress(Key.ControlLeft, emptyModifiers))
    handler.handleInput(InputEvent.KeyPress(Key.ShiftLeft, emptyModifiers))
    handler.handleInput(InputEvent.KeyPress(Key.AltLeft, emptyModifiers))

    handler.currentModifiers.ctrl shouldBe true
    handler.currentModifiers.shift shouldBe true
    handler.currentModifiers.alt shouldBe true

    handler.handleInput(InputEvent.KeyRelease(Key.ShiftLeft, emptyModifiers))
    handler.currentModifiers.ctrl shouldBe true
    handler.currentModifiers.shift shouldBe false
    handler.currentModifiers.alt shouldBe true

  "KeyHandler arrow key tracking" should "track arrow keys" in:
    val handler = OptiXKeyHandler(createDispatcher())

    handler.handleInput(InputEvent.KeyPress(Key.Left, emptyModifiers))
    handler.handleInput(InputEvent.KeyPress(Key.Up, emptyModifiers))

    // Arrow keys should return false (not consumed)
    handler.handleInput(InputEvent.KeyPress(Key.Right, emptyModifiers)) shouldBe false
    handler.handleInput(InputEvent.KeyRelease(Key.Right, emptyModifiers)) shouldBe false

  it should "track page up/down keys" in:
    val handler = OptiXKeyHandler(createDispatcher())
    handler.handleInput(InputEvent.KeyPress(Key.PageUp, emptyModifiers)) shouldBe false
    handler.handleInput(InputEvent.KeyRelease(Key.PageUp, emptyModifiers)) shouldBe false
    handler.handleInput(InputEvent.KeyPress(Key.PageDown, emptyModifiers)) shouldBe false

  "KeyHandler key return values" should "return false for modifier keys" in:
    val handler = OptiXKeyHandler(createDispatcher())
    handler.handleInput(InputEvent.KeyPress(Key.ControlLeft, emptyModifiers)) shouldBe false
    handler.handleInput(InputEvent.KeyPress(Key.ShiftLeft, emptyModifiers)) shouldBe false
    handler.handleInput(InputEvent.KeyPress(Key.AltLeft, emptyModifiers)) shouldBe false

  it should "return false for unknown keys" in:
    val handler = OptiXKeyHandler(createDispatcher())
    handler.handleInput(InputEvent.KeyPress(Key.Unknown(Keys.A), emptyModifiers)) shouldBe false
    handler.handleInput(InputEvent.KeyPress(Key.Unknown(Keys.SPACE), emptyModifiers)) shouldBe false

  it should "return false for arrow keys" in:
    val handler = OptiXKeyHandler(createDispatcher())
    handler.handleInput(InputEvent.KeyPress(Key.Left, emptyModifiers)) shouldBe false
    handler.handleInput(InputEvent.KeyPress(Key.Right, emptyModifiers)) shouldBe false
    handler.handleInput(InputEvent.KeyPress(Key.Up, emptyModifiers)) shouldBe false
    handler.handleInput(InputEvent.KeyPress(Key.Down, emptyModifiers)) shouldBe false

  "OptiXKeyHandler" should "not consume Q without Ctrl" in:
    val handler = OptiXKeyHandler(createDispatcher())
    handler.handleInput(InputEvent.KeyPress(Key.Unknown(Keys.Q), emptyModifiers)) shouldBe false

  it should "track rotation state" in:
    // Verify rotation tracking doesn't crash
    val handler = OptiXKeyHandler(createDispatcher())
    handler.handleInput(InputEvent.KeyPress(Key.Left, emptyModifiers))
    handler.handleInput(InputEvent.KeyPress(Key.Right, emptyModifiers))
    handler.handleInput(InputEvent.KeyRelease(Key.Left, emptyModifiers))
    handler.handleInput(InputEvent.KeyRelease(Key.Right, emptyModifiers))
    // No assertion - just checking no exception

  "Key sequence" should "handle rapid key presses" in:
    val handler = OptiXKeyHandler(createDispatcher())

    // Simulate rapid typing
    (1 to 100).foreach { _ =>
      handler.handleInput(InputEvent.KeyPress(Key.Left, emptyModifiers))
      handler.handleInput(InputEvent.KeyRelease(Key.Left, emptyModifiers))
    }
    // No assertion - checking for memory leaks or state corruption

  it should "handle overlapping key presses" in:
    val handler = OptiXKeyHandler(createDispatcher())

    handler.handleInput(InputEvent.KeyPress(Key.Left, emptyModifiers))
    handler.handleInput(InputEvent.KeyPress(Key.Right, emptyModifiers))
    handler.handleInput(InputEvent.KeyPress(Key.Up, emptyModifiers))
    handler.handleInput(InputEvent.KeyPress(Key.Down, emptyModifiers))

    handler.handleInput(InputEvent.KeyRelease(Key.Up, emptyModifiers))
    handler.handleInput(InputEvent.KeyRelease(Key.Left, emptyModifiers))
    handler.handleInput(InputEvent.KeyRelease(Key.Down, emptyModifiers))
    handler.handleInput(InputEvent.KeyRelease(Key.Right, emptyModifiers))
    // No assertion - checking state doesn't get corrupted

  it should "handle key up without prior key down" in:
    val handler = OptiXKeyHandler(createDispatcher())
    // This shouldn't crash
    handler.handleInput(InputEvent.KeyRelease(Key.Left, emptyModifiers)) shouldBe false
    handler.handleInput(InputEvent.KeyRelease(Key.ControlLeft, emptyModifiers)) shouldBe false
