package menger.engines

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// menger#103: a move that threw escaped the animation's render() and ended it.
class InPlaceMoveSuite extends AnyFlatSpec with Matchers:

  "InPlaceMove.attempt" should "return the moved state" in:
    InPlaceMove.attempt(1)(s => Some(s + 1)) shouldBe Some(2)

  it should "return None when the scene cannot take the frame" in:
    InPlaceMove.attempt(1)(_ => None) shouldBe None

  @SuppressWarnings(Array("org.wartremover.warts.Throw")) // a native update that fails
  private def failingMove(state: Int): Option[Int] =
    throw IllegalStateException(s"native update of $state failed")

  it should "return None, not throw, when the move throws, so the caller rebuilds" in:
    InPlaceMove.attempt(1)(failingMove) shouldBe None
