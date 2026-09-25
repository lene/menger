package menger.engines

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class WithPreviewSuite extends AnyFlatSpec with Matchers:

  "WithPreview.loopedT" should "follow the wall clock inside the time range" in:
    WithPreview.loopedT(elapsedSeconds = 2.5, startT = 0f, endT = 10f) shouldBe 2.5f

  it should "wrap around at the end of the range" in:
    WithPreview.loopedT(elapsedSeconds = 12.5, startT = 0f, endT = 10f) shouldBe 2.5f

  it should "offset by a non-zero start" in:
    WithPreview.loopedT(elapsedSeconds = 1.0, startT = 5f, endT = 7f) shouldBe 6f

  it should "stay at the start for an empty range" in:
    WithPreview.loopedT(elapsedSeconds = 3.0, startT = 4f, endT = 4f) shouldBe 4f
