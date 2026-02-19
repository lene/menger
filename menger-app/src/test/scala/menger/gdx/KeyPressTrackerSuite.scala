package menger.gdx

import menger.common.Key
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class KeyPressTrackerSuite extends AnyFlatSpec with Matchers:

  "KeyPressTracker" should "report all keys unpressed initially" in {
    val tracker = KeyPressTracker()
    tracker.isPressed(Key.Left) shouldBe false
    tracker.anyPressed shouldBe false
  }

  it should "track a pressed key" in {
    val tracker = KeyPressTracker()
    tracker.press(Key.Left)
    tracker.isPressed(Key.Left) shouldBe true
    tracker.anyPressed shouldBe true
  }

  it should "track a released key" in {
    val tracker = KeyPressTracker()
    tracker.press(Key.Left)
    tracker.release(Key.Left)
    tracker.isPressed(Key.Left) shouldBe false
    tracker.anyPressed shouldBe false
  }

  it should "track multiple keys independently" in {
    val tracker = KeyPressTracker()
    tracker.press(Key.Left)
    tracker.press(Key.Up)
    tracker.isPressed(Key.Left) shouldBe true
    tracker.isPressed(Key.Up) shouldBe true
    tracker.isPressed(Key.Right) shouldBe false
    tracker.anyPressed shouldBe true
  }

  it should "report anyPressed false only when all keys released" in {
    val tracker = KeyPressTracker()
    tracker.press(Key.Left)
    tracker.press(Key.Right)
    tracker.release(Key.Left)
    tracker.anyPressed shouldBe true
    tracker.release(Key.Right)
    tracker.anyPressed shouldBe false
  }

  it should "not throw on release of unpressed key" in {
    val tracker = KeyPressTracker()
    noException should be thrownBy tracker.release(Key.Down)
    tracker.isPressed(Key.Down) shouldBe false
  }

  it should "handle pressing the same key twice" in {
    val tracker = KeyPressTracker()
    tracker.press(Key.PageUp)
    tracker.press(Key.PageUp)
    tracker.isPressed(Key.PageUp) shouldBe true
    tracker.release(Key.PageUp)
    tracker.isPressed(Key.PageUp) shouldBe false
  }
