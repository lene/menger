package menger.input

import scala.collection.mutable.ListBuffer

import menger.RotationProjectionParameters
import menger.common.InputEvent
import menger.common.Key
import menger.common.ModifierState
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Tests for OptiXKeyHandler - OptiX keyboard input handler.
 */
class OptiXKeyHandlerSuite extends AnyFlatSpec with Matchers:

  class TestEventDispatcher extends EventDispatcher:
    private val events = ListBuffer[RotationProjectionParameters]()
    def getEvents: Seq[RotationProjectionParameters] = events.toSeq
    def clear(): Unit = events.clear()

    override def notifyObservers(event: RotationProjectionParameters): Unit =
      events += event

  "OptiXKeyHandler" should "handle key press events" in {
    val dispatcher = TestEventDispatcher()
    val handler = OptiXKeyHandler(dispatcher)

    val result = handler.handleInput(InputEvent.KeyPress(Key.Left, ModifierState()))
    result shouldBe false
  }

  it should "handle key release events" in {
    val dispatcher = TestEventDispatcher()
    val handler = OptiXKeyHandler(dispatcher)

    val result = handler.handleInput(InputEvent.KeyRelease(Key.Right, ModifierState()))
    result shouldBe false
  }

  it should "consume Escape key" in {
    val dispatcher = TestEventDispatcher()
    val handler = OptiXKeyHandler(dispatcher)

    // Escape quits the app - should be consumed
    // Note: Can't test Gdx.app.exit() without LibGDX runtime
    val result = handler.handleInput(InputEvent.KeyPress(Key.Escape, ModifierState()))
    result shouldBe true
  }

  it should "consume Ctrl+Q" in {
    val dispatcher = TestEventDispatcher()
    val handler = OptiXKeyHandler(dispatcher)

    val result = handler.handleInput(
      InputEvent.KeyPress(Key.Q, ModifierState(ctrl = true))
    )
    result shouldBe true
  }

  it should "not consume Q without Ctrl" in {
    val dispatcher = TestEventDispatcher()
    val handler = OptiXKeyHandler(dispatcher)

    val result = handler.handleInput(InputEvent.KeyPress(Key.Q, ModifierState()))
    result shouldBe false
  }

  it should "track all modifier keys" in {
    val dispatcher = TestEventDispatcher()
    val handler = OptiXKeyHandler(dispatcher)

    // Press modifiers
    handler.handleInput(InputEvent.KeyPress(Key.ControlLeft, ModifierState(ctrl = true)))
    handler.handleInput(InputEvent.KeyPress(Key.ShiftLeft, ModifierState(ctrl = true, shift = true)))
    handler.handleInput(InputEvent.KeyPress(Key.AltLeft, ModifierState(ctrl = true, shift = true, alt = true)))

    // Release modifiers
    handler.handleInput(InputEvent.KeyRelease(Key.ControlLeft, ModifierState(shift = true, alt = true)))
    handler.handleInput(InputEvent.KeyRelease(Key.ShiftLeft, ModifierState(alt = true)))
    handler.handleInput(InputEvent.KeyRelease(Key.AltLeft, ModifierState()))
  }

  it should "track right modifiers" in {
    val dispatcher = TestEventDispatcher()
    val handler = OptiXKeyHandler(dispatcher)

    handler.handleInput(InputEvent.KeyPress(Key.ControlRight, ModifierState(ctrl = true)))
    handler.handleInput(InputEvent.KeyPress(Key.ShiftRight, ModifierState(ctrl = true, shift = true)))
    handler.handleInput(InputEvent.KeyPress(Key.AltRight, ModifierState(ctrl = true, shift = true, alt = true)))
  }

  it should "track all arrow and page keys" in {
    val dispatcher = TestEventDispatcher()
    val handler = OptiXKeyHandler(dispatcher)

    handler.handleInput(InputEvent.KeyPress(Key.Left, ModifierState())) shouldBe false
    handler.handleInput(InputEvent.KeyPress(Key.Right, ModifierState())) shouldBe false
    handler.handleInput(InputEvent.KeyPress(Key.Up, ModifierState())) shouldBe false
    handler.handleInput(InputEvent.KeyPress(Key.Down, ModifierState())) shouldBe false
    handler.handleInput(InputEvent.KeyPress(Key.PageUp, ModifierState())) shouldBe false
    handler.handleInput(InputEvent.KeyPress(Key.PageDown, ModifierState())) shouldBe false
  }

  it should "handle rapid key sequences" in {
    val dispatcher = TestEventDispatcher()
    val handler = OptiXKeyHandler(dispatcher)

    (1 to 100).foreach { _ =>
      handler.handleInput(InputEvent.KeyPress(Key.Left, ModifierState()))
      handler.handleInput(InputEvent.KeyRelease(Key.Left, ModifierState()))
    }
  }

  it should "handle overlapping key presses" in {
    val dispatcher = TestEventDispatcher()
    val handler = OptiXKeyHandler(dispatcher)

    handler.handleInput(InputEvent.KeyPress(Key.Left, ModifierState()))
    handler.handleInput(InputEvent.KeyPress(Key.Right, ModifierState()))
    handler.handleInput(InputEvent.KeyPress(Key.Up, ModifierState()))
    handler.handleInput(InputEvent.KeyPress(Key.Down, ModifierState()))

    handler.handleInput(InputEvent.KeyRelease(Key.Up, ModifierState()))
    handler.handleInput(InputEvent.KeyRelease(Key.Left, ModifierState()))
    handler.handleInput(InputEvent.KeyRelease(Key.Down, ModifierState()))
    handler.handleInput(InputEvent.KeyRelease(Key.Right, ModifierState()))
  }

  it should "handle key up without prior key down" in {
    val dispatcher = TestEventDispatcher()
    val handler = OptiXKeyHandler(dispatcher)

    handler.handleInput(InputEvent.KeyRelease(Key.Left, ModifierState())) shouldBe false
    handler.handleInput(InputEvent.KeyRelease(Key.ControlLeft, ModifierState())) shouldBe false
  }

  it should "handle simultaneous modifiers" in {
    val dispatcher = TestEventDispatcher()
    val handler = OptiXKeyHandler(dispatcher)

    handler.handleInput(InputEvent.KeyPress(Key.ControlLeft, ModifierState(ctrl = true)))
    handler.handleInput(InputEvent.KeyPress(Key.ShiftLeft, ModifierState(ctrl = true, shift = true)))
    handler.handleInput(InputEvent.KeyPress(Key.AltLeft, ModifierState(ctrl = true, shift = true, alt = true)))

    // Release in different order
    handler.handleInput(InputEvent.KeyRelease(Key.ShiftLeft, ModifierState(ctrl = true, alt = true)))
    handler.handleInput(InputEvent.KeyRelease(Key.AltLeft, ModifierState(ctrl = true)))
    handler.handleInput(InputEvent.KeyRelease(Key.ControlLeft, ModifierState()))
  }
