package menger.input

import com.badlogic.gdx.Input.Buttons
import com.badlogic.gdx.Input.Keys
import menger.common.Key
import menger.common.MouseButton
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Tests for LibGDX key/button converters.
 * These tests require LibGDX runtime for Keys and Buttons constants.
 */
class LibGDXConvertersSuite extends AnyFlatSpec with Matchers:

  "LibGDXConverters.convertKey" should "convert modifier keys correctly" in {
    LibGDXConverters.convertKey(Keys.CONTROL_LEFT) shouldBe Key.ControlLeft
    LibGDXConverters.convertKey(Keys.CONTROL_RIGHT) shouldBe Key.ControlRight
    LibGDXConverters.convertKey(Keys.SHIFT_LEFT) shouldBe Key.ShiftLeft
    LibGDXConverters.convertKey(Keys.SHIFT_RIGHT) shouldBe Key.ShiftRight
    LibGDXConverters.convertKey(Keys.ALT_LEFT) shouldBe Key.AltLeft
    LibGDXConverters.convertKey(Keys.ALT_RIGHT) shouldBe Key.AltRight
  }

  it should "convert arrow keys correctly" in {
    LibGDXConverters.convertKey(Keys.LEFT) shouldBe Key.Left
    LibGDXConverters.convertKey(Keys.RIGHT) shouldBe Key.Right
    LibGDXConverters.convertKey(Keys.UP) shouldBe Key.Up
    LibGDXConverters.convertKey(Keys.DOWN) shouldBe Key.Down
  }

  it should "convert page keys correctly" in {
    LibGDXConverters.convertKey(Keys.PAGE_UP) shouldBe Key.PageUp
    LibGDXConverters.convertKey(Keys.PAGE_DOWN) shouldBe Key.PageDown
  }

  it should "convert special keys correctly" in {
    LibGDXConverters.convertKey(Keys.ESCAPE) shouldBe Key.Escape
    LibGDXConverters.convertKey(Keys.Q) shouldBe Key.Q
  }

  it should "handle unknown keys" in {
    val unknownKeyCode = 999999
    LibGDXConverters.convertKey(unknownKeyCode) shouldBe Key.Unknown(unknownKeyCode)
  }

  it should "be bidirectional for all known keys" in {
    val knownKeys = Seq(
      Keys.CONTROL_LEFT, Keys.CONTROL_RIGHT,
      Keys.SHIFT_LEFT, Keys.SHIFT_RIGHT,
      Keys.ALT_LEFT, Keys.ALT_RIGHT,
      Keys.LEFT, Keys.RIGHT, Keys.UP, Keys.DOWN,
      Keys.PAGE_UP, Keys.PAGE_DOWN,
      Keys.ESCAPE, Keys.Q
    )

    knownKeys.foreach { gdxKey =>
      val domainKey = LibGDXConverters.convertKey(gdxKey)
      domainKey should not be a[Key.Unknown]
    }
  }

  "LibGDXConverters.convertButton" should "convert mouse buttons correctly" in {
    LibGDXConverters.convertButton(Buttons.LEFT) shouldBe MouseButton.Left
    LibGDXConverters.convertButton(Buttons.RIGHT) shouldBe MouseButton.Right
    LibGDXConverters.convertButton(Buttons.MIDDLE) shouldBe MouseButton.Middle
  }

  it should "handle unknown buttons" in {
    val unknownButtonCode = 999999
    LibGDXConverters.convertButton(unknownButtonCode) shouldBe MouseButton.Unknown(unknownButtonCode)
  }

  it should "be bidirectional for all known buttons" in {
    val knownButtons = Seq(Buttons.LEFT, Buttons.RIGHT, Buttons.MIDDLE)

    knownButtons.foreach { gdxButton =>
      val domainButton = LibGDXConverters.convertButton(gdxButton)
      domainButton should not be a[MouseButton.Unknown]
    }
  }

  "LibGDXConverters.toGdxButton" should "convert Left to Buttons.LEFT" in:
    LibGDXConverters.toGdxButton(MouseButton.Left) shouldBe com.badlogic.gdx.Input.Buttons.LEFT

  it should "convert Right to Buttons.RIGHT" in:
    LibGDXConverters.toGdxButton(MouseButton.Right) shouldBe com.badlogic.gdx.Input.Buttons.RIGHT

  it should "pass through Unknown button code" in:
    LibGDXConverters.toGdxButton(MouseButton.Unknown(99)) shouldBe 99
