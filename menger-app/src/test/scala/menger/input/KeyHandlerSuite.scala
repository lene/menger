package menger.input

import menger.RotationProjectionParameters
import menger.common.{InputEvent, Key, ModifierState}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.collection.mutable.ListBuffer

/**
 * Pure domain tests for KeyHandler trait logic.
 * These tests require NO LibGDX runtime initialization.
 *
 * Uses immutable event collection for verification without vars in domain logic.
 */
class KeyHandlerSuite extends AnyFlatSpec with Matchers:

  /** Test implementation that collects events for verification */
  class RecordingKeyHandler extends KeyHandler:
    private val events = ListBuffer[(String, Key, ModifierState)]()

    def getEvents: Seq[(String, Key, ModifierState)] = events.toSeq
    def currentModifierState: ModifierState = modifierState

    override protected def handleKeyPress(key: Key, modifiers: ModifierState): Boolean =
      events += (("press", key, modifiers))
      key == Key.Escape  // Only Escape is consumed

    override protected def handleKeyRelease(key: Key, modifiers: ModifierState): Boolean =
      events += (("release", key, modifiers))
      false  // Never consumed

  "KeyHandler" should "receive and record key press events" in {
    val handler = RecordingKeyHandler()
    handler.handleInput(InputEvent.KeyPress(Key.Left, ModifierState()))

    handler.getEvents should contain (("press", Key.Left, ModifierState()))
  }

  it should "receive and record key release events" in {
    val handler = RecordingKeyHandler()
    handler.handleInput(InputEvent.KeyRelease(Key.Right, ModifierState()))

    handler.getEvents should contain (("release", Key.Right, ModifierState()))
  }

  it should "pass modifiers correctly to handlers" in {
    val handler = RecordingKeyHandler()
    val modifiers = ModifierState(ctrl = true, shift = true)

    handler.handleInput(InputEvent.KeyPress(Key.Q, modifiers))

    handler.getEvents.headOption match
      case Some(("press", Key.Q, mods)) =>
        mods.ctrl shouldBe true
        mods.shift shouldBe true
        mods.alt shouldBe false
      case _ => fail("Expected press event for Key.Q")
  }

  it should "return correct consumption status" in {
    val handler = RecordingKeyHandler()

    // Escape is consumed (returns true)
    val consumed = handler.handleInput(InputEvent.KeyPress(Key.Escape, ModifierState()))
    consumed shouldBe true

    // Left is not consumed (returns false)
    val notConsumed = handler.handleInput(InputEvent.KeyPress(Key.Left, ModifierState()))
    notConsumed shouldBe false
  }

  it should "track modifier state internally when processing events" in {
    val handler = RecordingKeyHandler()

    // Initial state
    handler.currentModifierState shouldBe ModifierState()

    // Press Ctrl
    handler.handleInput(InputEvent.KeyPress(Key.ControlLeft, ModifierState(ctrl = true)))
    handler.currentModifierState.ctrl shouldBe true

    // Press Shift
    handler.handleInput(InputEvent.KeyPress(Key.ShiftRight, ModifierState(ctrl = true, shift = true)))
    handler.currentModifierState.shift shouldBe true

    // Release Ctrl
    handler.handleInput(InputEvent.KeyRelease(Key.ControlLeft, ModifierState(shift = true)))
    handler.currentModifierState.ctrl shouldBe false
    handler.currentModifierState.shift shouldBe true
  }

  it should "ignore non-keyboard events" in {
    val handler = RecordingKeyHandler()

    val mouseEvent = InputEvent.MouseDown(
      menger.common.ScreenCoords(0, 0),
      menger.common.MouseButton.Left,
      0
    )
    val result = handler.handleInput(mouseEvent)

    result shouldBe false
    handler.getEvents shouldBe empty
  }

  it should "handle sequence of events correctly" in {
    val handler = RecordingKeyHandler()

    // Sequence of events
    handler.handleInput(InputEvent.KeyPress(Key.Left, ModifierState()))
    handler.handleInput(InputEvent.KeyPress(Key.Right, ModifierState()))
    handler.handleInput(InputEvent.KeyRelease(Key.Left, ModifierState()))
    handler.handleInput(InputEvent.KeyRelease(Key.Right, ModifierState()))

    val events = handler.getEvents
    events should have length 4
    events(0) shouldBe (("press", Key.Left, ModifierState()))
    events(1) shouldBe (("press", Key.Right, ModifierState()))
    events(2) shouldBe (("release", Key.Left, ModifierState()))
    events(3) shouldBe (("release", Key.Right, ModifierState()))
  }

  it should "handle all modifier keys correctly" in {
    val handler = RecordingKeyHandler()

    // Left modifiers
    handler.handleInput(InputEvent.KeyPress(Key.ControlLeft, ModifierState(ctrl = true)))
    handler.currentModifierState.ctrl shouldBe true

    handler.handleInput(InputEvent.KeyPress(Key.AltLeft, ModifierState(ctrl = true, alt = true)))
    handler.currentModifierState.alt shouldBe true

    handler.handleInput(InputEvent.KeyPress(Key.ShiftLeft, ModifierState(ctrl = true, alt = true, shift = true)))
    handler.currentModifierState.shift shouldBe true

    // Release left Ctrl
    handler.handleInput(InputEvent.KeyRelease(Key.ControlLeft, ModifierState(alt = true, shift = true)))
    handler.currentModifierState.ctrl shouldBe false

    // Press right Ctrl (should also work)
    handler.handleInput(InputEvent.KeyPress(Key.ControlRight, ModifierState(ctrl = true, alt = true, shift = true)))
    handler.currentModifierState.ctrl shouldBe true
  }
