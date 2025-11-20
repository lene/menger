package menger.objects.higher_d

import menger.common.Vector
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
    sponge.toString should include(s"level=${menger.common.float2string(sponge.level)}")

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

  "All vertices of TesseractSponge(1)" should "be inside or on the boundary of TesseractSponge(0)" in :
    val sponge = TesseractSponge(1)
    val boundingSponge = TesseractSponge(0)
    forAll(sponge.faces.flatMap(_.asSeq)) { v =>
      boundingSponge.isInSponge(v) should be (true)
    }

  "All vertices of TesseractSponge(2)" should "be inside or on the boundary of TesseractSponge(1)" in :
    pending
    val sponge = TesseractSponge(2)
    val boundingSponge = TesseractSponge(1)
    forAll(sponge.faces.flatMap(_.asSeq)) { v =>
      boundingSponge.isInSponge(v) should be (true)
    }

  "isInCube" should "return true for a point inside an axis-aligned cube" in:
    val sponge = TesseractSponge(0)
    val cubeVertices = sponge.faces.flatMap(_.asSeq)
    sponge.isInCube(Vector[4](0f, 0f, 0f, 0f), cubeVertices) should be (true)
    sponge.isInCube(Vector[4](0.25f, 0.25f, 0.25f, 0.25f), cubeVertices) should be (true)
    sponge.isInCube(Vector[4](-0.25f, -0.25f, -0.25f, -0.25f), cubeVertices) should be (true)

  it should "return true for a point on the boundary of an axis-aligned cube" in:
    val sponge = TesseractSponge(0)
    val cubeVertices = sponge.faces.flatMap(_.asSeq)
    sponge.isInCube(Vector[4](0.5f, 0f, 0f, 0f), cubeVertices) should be (true)
    sponge.isInCube(Vector[4](-0.5f, 0f, 0f, 0f), cubeVertices) should be (true)
    sponge.isInCube(Vector[4](0.5f, 0.5f, 0.5f, 0.5f), cubeVertices) should be (true)

  it should "return false for a point outside an axis-aligned cube" in:
    val sponge = TesseractSponge(0)
    val cubeVertices = sponge.faces.flatMap(_.asSeq)
    sponge.isInCube(Vector[4](1f, 0f, 0f, 0f), cubeVertices) should be (false)
    sponge.isInCube(Vector[4](-1f, 0f, 0f, 0f), cubeVertices) should be (false)
    sponge.isInCube(Vector[4](1f, 1f, 1f, 1f), cubeVertices) should be (false)

  it should "fail when given fewer than 16 vertices" in:
    val sponge = TesseractSponge(0)
    val tooFewVertices = Seq(
      Vector[4](-0.5f, -0.5f, -0.5f, -0.5f),
      Vector[4](0.5f, 0.5f, 0.5f, 0.5f)
    )
    an[IllegalArgumentException] should be thrownBy sponge.isInCube(Vector[4](0f, 0f, 0f, 0f), tooFewVertices)

  it should "fail when given more than 16 vertices" in:
    val sponge = TesseractSponge(0)
    val cubeVertices = sponge.faces.flatMap(_.asSeq)
    val tooManyVertices = cubeVertices ++ Seq(Vector[4](0f, 0f, 0f, 0f))
    an[IllegalArgumentException] should be thrownBy sponge.isInCube(Vector[4](0f, 0f, 0f, 0f), tooManyVertices)

  it should "fail when edges are not parallel to axes (rotated cube)" in:
    val sponge = TesseractSponge(0)
    // Create a "rotated" cube by having 3 distinct values in a dimension
    val rotatedVertices = Seq(
      Vector[4](0f, 0f, 0f, 0f),
      Vector[4](1f, 0f, 0f, 0f),
      Vector[4](0f, 1f, 0f, 0f),
      Vector[4](1f, 1f, 0f, 0f),
      Vector[4](0f, 0f, 1f, 0f),
      Vector[4](1f, 0f, 1f, 0f),
      Vector[4](0f, 1f, 1f, 0f),
      Vector[4](1f, 1f, 1f, 0f),
      Vector[4](0.5f, 0f, 0f, 1f), // This breaks axis-alignment in x dimension
      Vector[4](1f, 0f, 0f, 1f),
      Vector[4](0f, 1f, 0f, 1f),
      Vector[4](1f, 1f, 0f, 1f),
      Vector[4](0f, 0f, 1f, 1f),
      Vector[4](1f, 0f, 1f, 1f),
      Vector[4](0f, 1f, 1f, 1f),
      Vector[4](1f, 1f, 1f, 1f)
    )
    an[IllegalArgumentException] should be thrownBy sponge.isInCube(Vector[4](0f, 0f, 0f, 0f), rotatedVertices)

  it should "fail when vertices don't form a proper cube (only 1 distinct value in a dimension)" in:
    val sponge = TesseractSponge(0)
    // All vertices have the same x coordinate (degenerate in x dimension)
    val degenerateVertices = Seq(
      Vector[4](0f, 0f, 0f, 0f),
      Vector[4](0f, 1f, 0f, 0f),
      Vector[4](0f, 0f, 1f, 0f),
      Vector[4](0f, 1f, 1f, 0f),
      Vector[4](0f, 0f, 0f, 1f),
      Vector[4](0f, 1f, 0f, 1f),
      Vector[4](0f, 0f, 1f, 1f),
      Vector[4](0f, 1f, 1f, 1f),
      Vector[4](0f, 0f, 0f, 0f),
      Vector[4](0f, 1f, 0f, 0f),
      Vector[4](0f, 0f, 1f, 0f),
      Vector[4](0f, 1f, 1f, 0f),
      Vector[4](0f, 0f, 0f, 1f),
      Vector[4](0f, 1f, 0f, 1f),
      Vector[4](0f, 0f, 1f, 1f),
      Vector[4](0f, 1f, 1f, 1f)
    )
    an[IllegalArgumentException] should be thrownBy sponge.isInCube(Vector[4](0f, 0f, 0f, 0f), degenerateVertices)
