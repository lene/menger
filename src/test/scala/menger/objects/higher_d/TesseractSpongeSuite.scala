package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import org.scalatest.flatspec.AnyFlatSpec

class TesseractSpongeSuite extends AnyFlatSpec with RectMesh:

  trait Sponge:
    val sponge: TesseractSponge = TesseractSponge(1)

  "A TesseractSponge level 0" should "have 24 faces" in:
    val sponge = TesseractSponge(0)
    assert(sponge.faces.size == 24)

  "A TesseractSponge level < 0" should "be imposssible" in:
    assertThrows[IllegalArgumentException] {
      TesseractSponge(-1)
    }

  "A TesseractSponge level 1" should "have 48 times the number of a Tesseract's faces" in new Sponge:
    assert(sponge.faces.size == 48 * Tesseract().faces.size)

  it should "have no vertices with absolute value greater than 0.5" in new Sponge:
    assert(
      sponge.faces.forall(rect => rect.asSeq.forall(v => v.toArray.forall(_.abs <= 0.5)))
    )

  it should "have no face in the center of each face of the Tesseract" in new Sponge:
    assert(sponge.faces.forall(rect => !isCenterOfOriginalFace(rect)))

  it should "have no face around the removed center Tesseract" in new Sponge:
    assert(sponge.faces.forall(rect => !isCenterOfOriginalTesseract(rect)))


  def isCenterOfOriginalFace(face: RectVertices4D): Boolean =
    // A face is a center face if 2 of its coordinates are +/- 1/6 and the other 2 are 0.5
    face.asSeq.forall({ v =>
      val va = v.toArray
      va.count(_.abs == 0.5) == 2 && va.count(_.abs == 1 / 6f) == 2
    }
    )

  def isCenterOfOriginalTesseract(face: RectVertices4D): Boolean =
    // A face is a center face if all of its coordinates are +/- 1/6
    face.asSeq.forall { _.toArray.count(_.abs == 1 / 6f) == 4 }
