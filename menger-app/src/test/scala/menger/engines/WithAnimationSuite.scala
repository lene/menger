package menger.engines

import menger.ObjectSpec
import menger.Projection4DSpec
import menger.config.TAnimationConfig
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

  // --- GPU 4D projection animation fast-path eligibility helpers -----------

  private val tess0 = ObjectSpec(
    "tesseract", projection4D = Some(Projection4DSpec(rotXW = 0f, rotYW = 0f, rotZW = 0f))
  )
  private val tessRotated = ObjectSpec(
    "tesseract", projection4D = Some(Projection4DSpec(rotXW = 30f, rotYW = 0f, rotZW = 0f))
  )
  private val tessMoved = tess0.copy(x = 1f)

  "is4DOnlyTriangleMeshScene" should "accept a tesseract-only scene" in:
    WithAnimation.is4DOnlyTriangleMeshScene(List(tess0, tessRotated)) shouldBe true

  it should "reject a scene containing a non-projected type" in:
    val cube = ObjectSpec("cube")
    WithAnimation.is4DOnlyTriangleMeshScene(List(tess0, cube)) shouldBe false

  it should "reject an empty scene" in:
    WithAnimation.is4DOnlyTriangleMeshScene(List.empty) shouldBe false

  "specsDifferOnlyIn4DProjection" should "be true when only projection4D differs" in:
    WithAnimation.specsDifferOnlyIn4DProjection(List(tess0), List(tessRotated)) shouldBe true

  it should "be false when a non-projection field differs" in:
    WithAnimation.specsDifferOnlyIn4DProjection(List(tess0), List(tessMoved)) shouldBe false

  it should "be false when no spec actually changed" in:
    WithAnimation.specsDifferOnlyIn4DProjection(List(tess0), List(tess0)) shouldBe false

  it should "be false when lengths differ" in:
    WithAnimation.specsDifferOnlyIn4DProjection(List(tess0), List(tess0, tessRotated)) shouldBe false
