package menger.input

import com.badlogic.gdx.Input.Keys
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class KeyControllerSuite extends AnyFlatSpec with Matchers:

  "BaseKeyController modifier tracking" should "track Ctrl key press" in:
    val controller = OptiXKeyController()
    controller.ctrl shouldBe false

    controller.keyDown(Keys.CONTROL_LEFT)
    controller.ctrl shouldBe true

    controller.keyUp(Keys.CONTROL_LEFT)
    controller.ctrl shouldBe false

  it should "track Ctrl right key press" in:
    val controller = OptiXKeyController()
    controller.keyDown(Keys.CONTROL_RIGHT)
    controller.ctrl shouldBe true

    controller.keyUp(Keys.CONTROL_RIGHT)
    controller.ctrl shouldBe false

  it should "track Shift key press" in:
    val controller = OptiXKeyController()
    controller.shift shouldBe false

    controller.keyDown(Keys.SHIFT_LEFT)
    controller.shift shouldBe true

    controller.keyUp(Keys.SHIFT_LEFT)
    controller.shift shouldBe false

  it should "track Shift right key press" in:
    val controller = OptiXKeyController()
    controller.keyDown(Keys.SHIFT_RIGHT)
    controller.shift shouldBe true

  it should "track Alt key press" in:
    val controller = OptiXKeyController()
    controller.alt shouldBe false

    controller.keyDown(Keys.ALT_LEFT)
    controller.alt shouldBe true

    controller.keyUp(Keys.ALT_LEFT)
    controller.alt shouldBe false

  it should "track Alt right key press" in:
    val controller = OptiXKeyController()
    controller.keyDown(Keys.ALT_RIGHT)
    controller.alt shouldBe true

  it should "track multiple modifiers simultaneously" in:
    val controller = OptiXKeyController()

    controller.keyDown(Keys.CONTROL_LEFT)
    controller.keyDown(Keys.SHIFT_LEFT)
    controller.keyDown(Keys.ALT_LEFT)

    controller.ctrl shouldBe true
    controller.shift shouldBe true
    controller.alt shouldBe true

    controller.keyUp(Keys.SHIFT_LEFT)
    controller.ctrl shouldBe true
    controller.shift shouldBe false
    controller.alt shouldBe true

  "BaseKeyController arrow key tracking" should "track arrow keys" in:
    val controller = OptiXKeyController()

    controller.keyDown(Keys.LEFT)
    controller.keyDown(Keys.UP)

    // Arrow keys are tracked in rotatePressed map (protected)
    // We can verify by checking that keyDown returns false (not consumed)
    controller.keyDown(Keys.RIGHT) shouldBe false
    controller.keyUp(Keys.RIGHT) shouldBe false

  it should "track page up/down keys" in:
    val controller = OptiXKeyController()
    controller.keyDown(Keys.PAGE_UP) shouldBe false
    controller.keyUp(Keys.PAGE_UP) shouldBe false
    controller.keyDown(Keys.PAGE_DOWN) shouldBe false

  "BaseKeyController key return values" should "return false for modifier keys" in:
    val controller = OptiXKeyController()
    controller.keyDown(Keys.CONTROL_LEFT) shouldBe false
    controller.keyDown(Keys.SHIFT_LEFT) shouldBe false
    controller.keyDown(Keys.ALT_LEFT) shouldBe false

  it should "return false for unknown keys" in:
    val controller = OptiXKeyController()
    controller.keyDown(Keys.A) shouldBe false
    controller.keyDown(Keys.SPACE) shouldBe false

  it should "return false for arrow keys" in:
    val controller = OptiXKeyController()
    controller.keyDown(Keys.LEFT) shouldBe false
    controller.keyDown(Keys.RIGHT) shouldBe false
    controller.keyDown(Keys.UP) shouldBe false
    controller.keyDown(Keys.DOWN) shouldBe false

  "OptiXKeyController" should "not consume Q without Ctrl" in:
    val controller = OptiXKeyController()
    controller.keyDown(Keys.Q) shouldBe false

  it should "track rotation state via onRotationUpdate" in:
    // OptiXKeyController has empty onRotationUpdate, so this is a no-op
    // but we verify it doesn't crash
    val controller = OptiXKeyController()
    controller.keyDown(Keys.LEFT)
    controller.keyDown(Keys.RIGHT)
    controller.keyUp(Keys.LEFT)
    controller.keyUp(Keys.RIGHT)
    // No assertion - just checking no exception

  "Key sequence" should "handle rapid key presses" in:
    val controller = OptiXKeyController()

    // Simulate rapid typing
    (1 to 100).foreach { _ =>
      controller.keyDown(Keys.LEFT)
      controller.keyUp(Keys.LEFT)
    }
    // No assertion - checking for memory leaks or state corruption

  it should "handle overlapping key presses" in:
    val controller = OptiXKeyController()

    controller.keyDown(Keys.LEFT)
    controller.keyDown(Keys.RIGHT)
    controller.keyDown(Keys.UP)
    controller.keyDown(Keys.DOWN)

    controller.keyUp(Keys.UP)
    controller.keyUp(Keys.LEFT)
    controller.keyUp(Keys.DOWN)
    controller.keyUp(Keys.RIGHT)
    // No assertion - checking state doesn't get corrupted

  it should "handle key up without prior key down" in:
    val controller = OptiXKeyController()
    // This shouldn't crash
    controller.keyUp(Keys.LEFT) shouldBe false
    controller.keyUp(Keys.CONTROL_LEFT) shouldBe false
