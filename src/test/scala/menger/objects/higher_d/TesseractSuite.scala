package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import org.scalatest.flatspec.AnyFlatSpec

class TesseractSuite extends AnyFlatSpec:

  trait Fixture:
    val tesseract: Tesseract = Tesseract()

  "A Tesseract" should "have 16 vertices" in new Fixture:
    assert(tesseract.vertices.size == 16)

  it should "have vertex coordinates all at +/-0.5" in new Fixture:
    assert(tesseract.vertices.forall(v => v.toArray.forall(_.abs == 0.5)))

  it should "have vertices scale with tesseract size" in:
    Seq(2.0f, 10.0f, 1e8f, 0.5f, 1e-8f).foreach { size =>
      assert(Tesseract(2 * size).vertices.forall(v => v.toArray.forall(_.abs == size)))
    }

  it should "have 24 faces" in new Fixture:
    assert(tesseract.faceIndices.size == 24)

  it should "have correct first face" in new Fixture:
    assert(
      tesseract.faces.head == Face4D(
        Vector4(-0.5,-0.5,-0.5,-0.5), Vector4(-0.5,-0.5,-0.5, 0.5),
        Vector4(-0.5,-0.5, 0.5, 0.5), Vector4(-0.5,-0.5, 0.5,-0.5)
      )
    )

  it should "have correct last face" in new Fixture:
    assert(
      tesseract.faces.last == Face4D(
        Vector4( 0.5, 0.5,-0.5,-0.5), Vector4( 0.5, 0.5,-0.5, 0.5),
        Vector4( 0.5, 0.5, 0.5, 0.5), Vector4( 0.5, 0.5, 0.5,-0.5)
      )
    )

  it should "have 32 edges" in new Fixture:
    assert(tesseract.edges.size == 32)

  it should "have only edges of unit length" in new Fixture:
    val edgeLengths: Seq[Float] = tesseract.edges.map { case (a, b) => a.dst(b) }
    assert(edgeLengths.forall { _ == 1.0f })

  "Face4D normal" should "point into +xy for the first Face4D" in new Fixture:
    withClue(s"${tesseract.faces.head.plane}:\n${tesseract.faces.head.normals}") {
      assert(tesseract.faces.head.normals.toSet == Set(Vector4.X, Vector4.Y))
    }

  it should "point into -xy for the 2nd Face4D" in new Fixture:
    withClue(s"Face in ${tesseract.faces(1).plane} plane has ${tesseract.faces(1).plane.neg} normals:\n${tesseract.faces(1).normals}") {
      assert(tesseract.faces(1).normals.toSet == tesseract.faces(1).plane.units.toSet)
    }
