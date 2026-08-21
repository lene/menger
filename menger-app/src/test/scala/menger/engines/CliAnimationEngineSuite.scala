package menger.engines

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CliAnimationEngineSuite extends AnyFlatSpec with Matchers:

  "CliAnimationEngine.formatSaveName" should "return None when no save pattern is given" in:
    CliAnimationEngine.formatSaveName(None, 3) shouldBe None

  it should "format the pattern with the frame number when a save pattern is given" in:
    CliAnimationEngine.formatSaveName(Some("frame-%d.png"), 3) shouldBe Some("frame-3.png")
