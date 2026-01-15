package menger.input

import menger.common.{InputEvent, Key, MouseButton, ModifierState, ScreenCoords}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Pure domain tests for InputEvent sealed trait hierarchy.
 * These tests require NO LibGDX runtime initialization.
 */
class InputEventSuite extends AnyFlatSpec with Matchers:

  "InputEvent.KeyPress" should "be created with correct values" in {
    val event = InputEvent.KeyPress(Key.Left, ModifierState(shift = true))
    event.key shouldBe Key.Left
    event.modifiers.shift shouldBe true
    event.modifiers.ctrl shouldBe false
    event.modifiers.alt shouldBe false
  }

  "InputEvent.KeyRelease" should "be created with correct values" in {
    val event = InputEvent.KeyRelease(Key.Escape, ModifierState(ctrl = true))
    event.key shouldBe Key.Escape
    event.modifiers.ctrl shouldBe true
    event.modifiers.shift shouldBe false
  }

  "InputEvent.MouseDown" should "track position, button, and pointer" in {
    val event = InputEvent.MouseDown(ScreenCoords(100, 200), MouseButton.Left, 0)
    event.position.x shouldBe 100
    event.position.y shouldBe 200
    event.button shouldBe MouseButton.Left
    event.pointer shouldBe 0
  }

  "InputEvent.MouseUp" should "track position, button, and pointer" in {
    val event = InputEvent.MouseUp(ScreenCoords(50, 75), MouseButton.Right, 1)
    event.position.x shouldBe 50
    event.position.y shouldBe 75
    event.button shouldBe MouseButton.Right
    event.pointer shouldBe 1
  }

  "InputEvent.MouseDrag" should "track position, pointer, and button" in {
    val event = InputEvent.MouseDrag(ScreenCoords(150, 250), 0, MouseButton.Left)
    event.position.x shouldBe 150
    event.position.y shouldBe 250
    event.pointer shouldBe 0
    event.button shouldBe MouseButton.Left
  }

  "InputEvent.ScrollEvent" should "track scroll amounts" in {
    val event = InputEvent.ScrollEvent(1.5f, -2.0f)
    event.amountX shouldBe 1.5f
    event.amountY shouldBe -2.0f
  }

  "ModifierState" should "update correctly with withCtrl" in {
    val initial = ModifierState()
    val withCtrl = initial.withCtrl(true)

    withCtrl.ctrl shouldBe true
    withCtrl.shift shouldBe false
    withCtrl.alt shouldBe false
  }

  it should "update correctly with withShift" in {
    val initial = ModifierState(ctrl = true)
    val withShift = initial.withShift(true)

    withShift.ctrl shouldBe true
    withShift.shift shouldBe true
    withShift.alt shouldBe false
  }

  it should "update correctly with withAlt" in {
    val initial = ModifierState(shift = true)
    val withAlt = initial.withAlt(true)

    withAlt.ctrl shouldBe false
    withAlt.shift shouldBe true
    withAlt.alt shouldBe true
  }

  it should "chain updates correctly" in {
    val initial = ModifierState()
    val updated = initial.withCtrl(true).withShift(true).withAlt(true)

    updated.ctrl shouldBe true
    updated.shift shouldBe true
    updated.alt shouldBe true
  }

  "Key enum" should "have all expected modifier keys" in {
    Key.ControlLeft shouldBe Key.ControlLeft
    Key.ControlRight shouldBe Key.ControlRight
    Key.ShiftLeft shouldBe Key.ShiftLeft
    Key.ShiftRight shouldBe Key.ShiftRight
    Key.AltLeft shouldBe Key.AltLeft
    Key.AltRight shouldBe Key.AltRight
  }

  it should "have all expected arrow keys" in {
    Key.Left shouldBe Key.Left
    Key.Right shouldBe Key.Right
    Key.Up shouldBe Key.Up
    Key.Down shouldBe Key.Down
  }

  it should "have all expected page keys" in {
    Key.PageUp shouldBe Key.PageUp
    Key.PageDown shouldBe Key.PageDown
  }

  it should "have all expected special keys" in {
    Key.Escape shouldBe Key.Escape
    Key.Q shouldBe Key.Q
  }

  it should "support unknown keys with code" in {
    val unknown = Key.Unknown(999)
    unknown match
      case Key.Unknown(code) => code shouldBe 999
      case _ => fail("Expected Unknown key")
  }

  "MouseButton enum" should "have all expected buttons" in {
    MouseButton.Left shouldBe MouseButton.Left
    MouseButton.Right shouldBe MouseButton.Right
    MouseButton.Middle shouldBe MouseButton.Middle
  }

  it should "support unknown buttons with code" in {
    val unknown = MouseButton.Unknown(999)
    unknown match
      case MouseButton.Unknown(code) => code shouldBe 999
      case _ => fail("Expected Unknown button")
  }

  "ScreenCoords" should "store coordinates correctly" in {
    val coords = ScreenCoords(640, 480)
    coords.x shouldBe 640
    coords.y shouldBe 480
  }
