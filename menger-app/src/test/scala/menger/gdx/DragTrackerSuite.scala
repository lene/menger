package menger.gdx

import menger.common.ScreenCoords
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DragTrackerSuite extends AnyFlatSpec with Matchers:

  "DragTracker" should "start at (0,0)" in {
    val tracker = DragTracker()
    tracker.origin shouldBe ScreenCoords(0, 0)
  }

  it should "update origin on start" in {
    val tracker = DragTracker()
    tracker.start(ScreenCoords(100, 200))
    tracker.origin shouldBe ScreenCoords(100, 200)
  }

  it should "update origin on each subsequent start" in {
    val tracker = DragTracker()
    tracker.start(ScreenCoords(10, 20))
    tracker.start(ScreenCoords(30, 40))
    tracker.origin shouldBe ScreenCoords(30, 40)
  }
