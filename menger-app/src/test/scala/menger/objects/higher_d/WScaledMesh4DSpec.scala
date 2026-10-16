package menger.objects.higher_d

import menger.ObjectSpec
import menger.Projection4DSpec
import menger.engines.WithAnimation
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** menger#65 (usability review 2026-10, session 3, F81): a w-scale applied before rotation and
  * projection, so a scene can grow a tesseract out of a flat cube. */
class WScaledMesh4DSpec extends AnyFlatSpec with Matchers:

  private val tesseract = Tesseract(size = 2f)

  "WScaledMesh4D" should "flatten a tesseract to w = 0 at scale 0" in:
    val flat = WScaledMesh4D(tesseract, 0f)
    flat.vertices.map(_(3)).toSet shouldBe Set(0f)
    flat.faces.flatMap(_.asSeq).map(_(3)).toSet shouldBe Set(0f)

  it should "scale only w" in:
    val half = WScaledMesh4D(tesseract, 0.5f)
    half.vertices.map(v => (v(0), v(1), v(2), v(3))) shouldBe
      tesseract.vertices.map(v => (v(0), v(1), v(2), v(3) * 0.5f))

  it should "keep the face count and vertices per face" in:
    val flat = WScaledMesh4D(tesseract, 0f)
    (flat.faces.size, flat.vertsPerFace) shouldBe (tesseract.faces.size, tesseract.vertsPerFace)

  "WScaledMesh4D.of" should "return the mesh itself at scale 1" in:
    WScaledMesh4D.of(tesseract, 1f) shouldBe theSameInstanceAs(tesseract)

  "Projection4DSpec" should "reject a negative w-scale" in:
    an[IllegalArgumentException] should be thrownBy Projection4DSpec(wScale = -0.1f)

  "ObjectSpec.parse" should "read the w-scale keyword" in:
    ObjectSpec.parse("type=tesseract:w-scale=0.25").map(_.wScale) shouldBe Right(0.25f)

  it should "reject a negative w-scale" in:
    ObjectSpec.parse("type=tesseract:w-scale=-1").isLeft shouldBe true

  "WithAnimation.specsDifferOnlyIn4DProjection" should "not count a w-scale change as a view change" in:
    val spec = ObjectSpec("tesseract")
    val scaled = spec.copy(projection4D = Some(Projection4DSpec(wScale = 0.5f)))
    WithAnimation.specsDifferOnlyIn4DProjection(List(spec), List(scaled)) shouldBe false
    val turned = spec.copy(projection4D = Some(Projection4DSpec(rotXW = 40f)))
    WithAnimation.specsDifferOnlyIn4DProjection(List(spec), List(turned)) shouldBe true
