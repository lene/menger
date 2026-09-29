package menger.objects.higher_d

import menger.common.Vector
import org.scalatest.Inspectors.forAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TesseractSpongeSuite extends AnyFlatSpec with Matchers:

  trait Sponge:
    val sponge: TesseractSponge = TesseractSponge(1)

  "A TesseractSponge level 0" should "have 24 faces" in:
    val sponge = TesseractSponge(0)
    sponge.faces should have size 24

  "A TesseractSponge level < 0" should "be impossible" in:
    an[IllegalArgumentException] should be thrownBy TesseractSponge(-1)

  // Usability review 2026-09 (F27): `size` was ignored and every sponge had unit size.
  "A TesseractSponge with size 2.5" should "be 2.5 times as large as a unit sponge" in:
    def maxNorm(s: TesseractSponge): Float = s.vertices.map(_.len).max
    maxNorm(TesseractSponge(1, size = 2.5f)) shouldBe (maxNorm(TesseractSponge(1)) * 2.5f +- 1e-4f)

  it should "keep the same number of faces" in:
    TesseractSponge(1, size = 2.5f).faces should have size TesseractSponge(1).faces.size

  "A TesseractSponge size <= 0" should "be impossible" in:
    an[IllegalArgumentException] should be thrownBy TesseractSponge(1, size = 0f)

  // Usability review 2026-09, session 2 (F55): every sub-tesseract emitted all 24 of its faces,
  // so a face shared by neighbours existed 2-4 times at the same place (level 1 had 48 * 24 =
  // 1152 faces for 768 distinct ones), and glass refracted at every copy.
  "A TesseractSponge" should "have no two faces at the same place" in:
    Seq(1, 2).foreach { level =>
      val keys = TesseractSponge(level).faces.map(faceKey)
      withClue(s"level $level: ") { keys.distinct.size shouldBe keys.size }
    }

  Seq(1, 2).foreach { level =>
    it should s"consist of exactly the faces between a filled and an empty hypercube (level $level)" in:
      TesseractSponge(level).faces.map(faceKey).toSet shouldBe boundaryFaceKeys(level)
  }

  private type Key = Seq[(Int, Int, Int, Int)]
  private val KeyScale = 1e4f

  private def keyOf(points: Seq[Seq[Float]]): Key =
    points.map(p => (
      math.round(p(0) * KeyScale), math.round(p(1) * KeyScale),
      math.round(p(2) * KeyScale), math.round(p(3) * KeyScale)
    )).sorted

  private def faceKey(face: Face4D[4]): Key = keyOf(face.asSeq.map(v => (0 until 4).map(v(_))))

  /** Independent oracle: the unit sponge as a grid of 3^level cells per axis, a cell filled
    * unless, at some base-3 digit, two or more of its coordinates are "middle" (digit 1). A
    * 2-face of the grid is on the surface iff the four cells around it are neither all filled
    * nor all empty. */
  private def boundaryFaceKeys(level: Int): Set[Key] =
    val n = math.pow(3, level).toInt
    def filled(cell: Seq[Int]): Boolean =
      cell.forall(i => i >= 0 && i < n) &&
        (0 until level).forall { k =>
          cell.count(i => (i / math.pow(3, k).toInt) % 3 == 1) < 2
        }
    val axes = 0 until 4
    val faces = for
      a <- axes; b <- axes if a < b
      p <- (0 to n).flatMap(x => (0 to n).flatMap(y => (0 to n).flatMap(z => (0 to n).map(w =>
        IndexedSeq(x, y, z, w)))))
      if p(a) < n && p(b) < n
    yield
      val Seq(c, d) = axes.filterNot(i => i == a || i == b)
      val around = for dc <- Seq(-1, 0); dd <- Seq(-1, 0) yield
        p.updated(c, p(c) + dc).updated(d, p(d) + dd)
      val filledCount = around.count(filled)
      val corners = Seq((0, 0), (1, 0), (1, 1), (0, 1)).map { (da, db) =>
        p.updated(a, p(a) + da).updated(b, p(b) + db).map(i => -0.5f + i.toFloat / n)
      }
      (keyOf(corners), filledCount)
    faces.collect { case (key, count) if count > 0 && count < 4 => key }.toSet

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

  private def isCenterOfOriginalFace(face: Face4D[4]): Boolean =
    // A face is a center face if 2 of its coordinates are +/- 1/6 and the other 2 are 0.5
    face.asSeq.forall({ v =>
      v.count(_.abs == 0.5) == 2 && v.count(_.abs == 1 / 6f) == 2
    })

  private def isCenterOfOriginalTesseract(face: Face4D[4]): Boolean =
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

  "TesseractSponge2 level 1 vertices" should
    "all lie within TesseractSponge level 1 region" in:
      pending  // isInSponge not yet implemented for level > 0
      val surface = TesseractSponge2(1)
      val volume  = TesseractSponge(1)
      forAll(surface.faces.flatMap(_.asSeq)) { v =>
        volume.isInSponge(v) should be(true)
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
