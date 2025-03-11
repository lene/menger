package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Inspectors.forAll

class TesseractSuite extends AnyFlatSpec with Matchers:

  val tesseract: Tesseract = Tesseract()

  "A Tesseract" should "have 16 vertices" in:
    tesseract.vertices should have size 16

  it should "have vertex coordinates all at +/-0.5" in:
    forAll(tesseract.vertices) {v => forAll(v.toArray) {s => s.abs == 0.5} }

  it should "have vertices scale with tesseract size" in:
    forAll(Seq(2.0f, 10.0f, 1e8f, 0.5f, 1e-8f)) { size =>
      forAll(Tesseract(2 * size).vertices) { v => forAll(v.toArray) { s => s.abs == size }}
    }

  it should "have 24 faces" in:
    tesseract.faceIndices should have size 24

  it should "have correct first face" in:
    tesseract.faces.head should be (Face4D(
      Vector4(-0.5,-0.5,-0.5,-0.5), Vector4(-0.5,-0.5,-0.5, 0.5),
      Vector4(-0.5,-0.5, 0.5, 0.5), Vector4(-0.5,-0.5, 0.5,-0.5)
    ))

  it should "have correct last face" in:
    tesseract.faces.last should be (Face4D(
      Vector4( 0.5, 0.5, 0.5, 0.5), Vector4( 0.5, 0.5,-0.5, 0.5),
      Vector4( 0.5, 0.5,-0.5,-0.5), Vector4( 0.5, 0.5, 0.5,-0.5)
    ))

  it should "have 32 edges" in:
    tesseract.edges should have size 32

  it should "have only edges of unit length" in:
    val edgeLengths: Seq[Float] = tesseract.edges.map { case (a, b) => a.dst(b) }
    edgeLengths should contain only 1.0f

  "Face4D normal" should "point into +xy for the first Face4D" in:
    withClue(faceClue(tesseract.faces.head)) {
      tesseract.faces.head.normals should contain only (Vector4.X, Vector4.Y)
    }

  it should "point into +xz for the 2nd Face4D" in:
    withClue(faceClue(tesseract.faces(1))) {
      tesseract.faces(1).normals should contain only (Vector4.X, Vector4.Z)
    }

  it should "point into +yz for the 3rd Face4D" in:
    withClue(faceClue(tesseract.faces(2))) {
      tesseract.faces(2).normals should contain only (Vector4.Y, Vector4.Z)
    }

  def faceClue(face: Face4D): String =
    s"Face in ${face.plane} plane has ${face.plane.neg} normals:\n${face.normals}"
