package menger.engines

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class WithAnimationSuite extends AnyFlatSpec with Matchers:

  "TAnimationConfig" should "compute tForFrame correctly for 10 frames over [0,1]" in:
    val config =
      TAnimationConfig(startT = 0f, endT = 1f, frames = 10, savePattern = "frame_%04d.png")
    config.tForFrame(0) shouldBe 0f
    config.tForFrame(9) shouldBe (1f +- 0.001f)

  it should "compute tForFrame correctly for 5 frames over [0.5, 1.5]" in:
    val config = TAnimationConfig(startT = 0.5f, endT = 1.5f, frames = 5, savePattern = "f_%d.png")
    config.tForFrame(0) shouldBe (0.5f +- 0.001f)
    config.tForFrame(4) shouldBe (1.5f +- 0.001f)

  "frame completion predicate" should "be true when frame >= animConfig.frames" in:
    val config = TAnimationConfig(startT = 0f, endT = 1f, frames = 3, savePattern = "f_%d.png")
    (3 >= config.frames) shouldBe true
    (2 >= config.frames) shouldBe false

  "save name formatting" should "format frame index with %04d pattern" in:
    val pattern = "animation_%04d.png"
    String.format(pattern, Integer.valueOf(0))   shouldBe "animation_0000.png"
    String.format(pattern, Integer.valueOf(42))  shouldBe "animation_0042.png"
    String.format(pattern, Integer.valueOf(999)) shouldBe "animation_0999.png"

  it should "format frame index with %d pattern" in:
    val pattern = "frame_%d.png"
    String.format(pattern, Integer.valueOf(7)) shouldBe "frame_7.png"
