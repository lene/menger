package menger.engines

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TAnimationConfigSuite extends AnyFlatSpec with Matchers:

  "TAnimationConfig" should "interpolate t linearly across frames" in:
    val config = TAnimationConfig(0f, 1f, 11, "frame_%04d.png")
    config.tForFrame(0) shouldBe 0f
    config.tForFrame(5) shouldBe 0.5f +- 0.001f
    config.tForFrame(10) shouldBe 1f

  it should "handle single frame" in:
    val config = TAnimationConfig(0.5f, 2f, 1, "frame_%04d.png")
    config.tForFrame(0) shouldBe 0.5f

  it should "handle equal start and end" in:
    val config = TAnimationConfig(1f, 1f, 5, "frame_%04d.png")
    config.tForFrame(0) shouldBe 1f
    config.tForFrame(4) shouldBe 1f

  it should "handle negative ranges" in:
    val config = TAnimationConfig(-1f, 1f, 3, "frame_%04d.png")
    config.tForFrame(0) shouldBe -1f
    config.tForFrame(1) shouldBe 0f +- 0.001f
    config.tForFrame(2) shouldBe 1f

  it should "handle reversed ranges (endT < startT)" in:
    val config = TAnimationConfig(1f, 0f, 3, "frame_%04d.png")
    config.tForFrame(0) shouldBe 1f
    config.tForFrame(1) shouldBe 0.5f +- 0.001f
    config.tForFrame(2) shouldBe 0f

  it should "work with a 2-pi range" in:
    val twoPi = (2 * math.Pi).toFloat
    val config = TAnimationConfig(0f, twoPi, 100, "orbit_%04d.png")
    config.tForFrame(0) shouldBe 0f
    config.tForFrame(99) shouldBe twoPi +- 0.001f

  it should "reject zero frames" in:
    an[IllegalArgumentException] should be thrownBy
      TAnimationConfig(0f, 1f, 0, "frame_%04d.png")

  it should "reject negative frames" in:
    an[IllegalArgumentException] should be thrownBy
      TAnimationConfig(0f, 1f, -1, "frame_%04d.png")

  it should "reject save pattern without %" in:
    an[IllegalArgumentException] should be thrownBy
      TAnimationConfig(0f, 1f, 10, "output.png")
