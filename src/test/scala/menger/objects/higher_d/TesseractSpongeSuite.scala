package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Inspectors.forAll

class TesseractSpongeSuite extends AnyFlatSpec with RectMesh with Matchers:

  trait Sponge:
    val sponge: TesseractSponge = TesseractSponge(1)

  "A TesseractSponge level 0" should "have 24 faces" in:
    val sponge = TesseractSponge(0)
    sponge.faces should have size 24

  "A TesseractSponge level < 0" should "be impossible" in:
    an[IllegalArgumentException] should be thrownBy TesseractSponge(-1)

  "A TesseractSponge level 1" should "have 48 times the number of a Tesseract's faces" in new Sponge:
    sponge.faces should have size 48 * Tesseract().faces.size

  it should "have no vertices with absolute value greater than 0.5" in new Sponge:
    forAll(sponge.faces) { rect => forAll(rect.asSeq) { v => v.forall(_.abs <= 0.5) } }

  it should "have no face in the center of each face of the Tesseract" in new Sponge:
    forAll(sponge.faces) { rect => !isCenterOfOriginalFace(rect) }

  it should "have no face around the removed center Tesseract" in new Sponge:
    forAll(sponge.faces) { rect => !isCenterOfOriginalTesseract(rect) }

  "toString" should "return the class name" in new Sponge:
    sponge.toString should include("TesseractSponge")

  it should "contain the sponge level" in new Sponge:
    sponge.toString should include(s"level=${menger.objects.float2string(sponge.level)}")

  it should "contain the number of faces" in new Sponge:
    sponge.toString should include(s"${sponge.faces.size} faces")

  private def isCenterOfOriginalFace(face: Face4D): Boolean =
    // A face is a center face if 2 of its coordinates are +/- 1/6 and the other 2 are 0.5
    face.asSeq.forall({ v =>
      v.count(_.abs == 0.5) == 2 && v.count(_.abs == 1 / 6f) == 2
    })

  private def isCenterOfOriginalTesseract(face: Face4D): Boolean =
    // A face is a center face if all of its coordinates are +/- 1/6
    face.asSeq.forall { _.count(_.abs == 1 / 6f) == 4 }

  "fractional level 0.5" should "instantiate" in:
    val sponge = TesseractSponge(0.5f)
    sponge.level shouldBe 0.5f

  it should "use floor for face generation" in:
    val sponge = TesseractSponge(0.5f)
    val level0 = TesseractSponge(0f)
    sponge.faces should have size level0.faces.size
