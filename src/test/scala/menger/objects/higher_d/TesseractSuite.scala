package menger.objects.higher_d

import menger.common.Vector
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Inspectors.forAll

class TesseractSuite extends AnyFlatSpec with Matchers:

  val tesseract: Tesseract = Tesseract()

  "A Tesseract" should "have 16 vertices" in:
    tesseract.vertices should have size 16

  it should "have vertex coordinates all at +/-0.5" in:
    forAll(tesseract.vertices) {v => v.forall { s => s.abs == 0.5} }

  it should "have vertices scale with tesseract size" in:
    forAll(Seq(2.0f, 10.0f, 1e8f, 0.5f, 1e-8f)) { size =>
      forAll(Tesseract(size).vertices) { v => v.forall { s => s.abs == size / 2 } }
    }

  it should "be centered at the origin" in :
    val center = tesseract.vertices.reduce((a, b) => a + b) / tesseract.vertices.size.toFloat
    center should be (Vector[4](0, 0, 0, 0))

  it should "have 24 faces" in:
    tesseract.faceIndices should have size 24

  it should "have correct first face" in:
    tesseract.faces.head should be (Face4D(
      Vector[4](-0.5,-0.5,-0.5,-0.5), Vector[4](-0.5,-0.5,-0.5, 0.5),
      Vector[4](-0.5,-0.5, 0.5, 0.5), Vector[4](-0.5,-0.5, 0.5,-0.5)
    ))

  it should "have correct last face" in:
    tesseract.faces.last should be (Face4D(
      Vector[4]( 0.5, 0.5, 0.5, 0.5), Vector[4]( 0.5, 0.5,-0.5, 0.5),
      Vector[4]( 0.5, 0.5,-0.5,-0.5), Vector[4]( 0.5, 0.5, 0.5,-0.5)
    ))

  it should "have 32 edges" in:
    tesseract.edges should have size 32

  it should "have only edges of unit length" in:
    val edgeLengths: Seq[Float] = tesseract.edges.map { case (a, b) => a.dst(b) }
    edgeLengths should contain only 1.0f

  it should "have all faces lying in specific planes" in :
    val planes = Set(Plane.xy, Plane.xz, Plane.xw, Plane.yz, Plane.yw, Plane.zw)
    forAll(tesseract.faces) { face => planes should contain(face.plane) }

  it should "have 4 faces for each of the 6 possible planes" in :
    val facePlanes = tesseract.faces.map(_.plane)
    val planeCounts = facePlanes.groupBy(identity).view.mapValues(_.size).toMap
    planeCounts.values should contain only 4
    planeCounts.size should be(6)

  "Face4D normal" should "point into +xy for the first Face4D" in:
    withClue(faceClue(tesseract.faces.head)) {
      tesseract.faces.head.normals should contain only (Vector.X, Vector.Y)
    }

  it should "point into +xz for the 2nd Face4D" in:
    withClue(faceClue(tesseract.faces(1))) {
      tesseract.faces(1).normals should contain only (Vector.X, Vector.Z)
    }

  it should "point into +yz for the 3rd Face4D" in:
    withClue(faceClue(tesseract.faces(2))) {
      tesseract.faces(2).normals should contain only (Vector.Y, Vector.Z)
    }

  def faceClue(face: Face4D): String =
    s"Face in ${face.plane} plane has ${face.plane.neg} normals:\n${face.normals}"
