package menger.objects.higher_d

import menger.common.Vector
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Inspectors.forAll

class Face4DSuite extends AnyFlatSpec with RectMesh with Matchers:

  private val seqXY = Seq(Vector[4](-1, -1, 0, 0), Vector[4](1, -1, 0, 0), Vector[4](1, 1, 0, 0), Vector[4](-1, 1, 0, 0))
  private val faceXY = Face4D(seqXY)
  private val faceXZ = Face4D(Vector[4](-1, 0, -1, 0), Vector[4](1, 0, -1, 0), Vector[4](1, 0, 1, 0), Vector[4](-1, 0, 1, 0))
  private val faceXW = Face4D(Vector[4](-1, 0, 0, -1), Vector[4](1, 0, 0, -1), Vector[4](1, 0, 0, 1), Vector[4](-1, 0, 0, 1))
  private val faceYZ = Face4D(Vector[4](0, -1, -1, 0), Vector[4](0, 1, -1, 0), Vector[4](0, 1, 1, 0), Vector[4](0, -1, 1, 0))
  private val faceYW = Face4D(Vector[4](0, -1, 0, -1), Vector[4](0, 1, 0, -1), Vector[4](0, 1, 0, 1), Vector[4](0, -1, 0, 1))
  private val faceZW = Face4D(Vector[4](0, 0, -1, -1), Vector[4](0, 0, 1, -1), Vector[4](0, 0, 1, 1), Vector[4](0, 0, -1, 1))

  "Face4D area" should "be correct in xy plane" in:
    faceXY.area should be (4.0f)

  Seq(faceXZ, faceXW, faceYZ, faceYW, faceZW).foreach { face =>
    it should s"be correct in ${face.plane} plane" in:
      face.area shouldBe 4.0f
  }

  "a Face4D" should "convert to tuple and sequence correctly" in :
    faceXY.asTuple shouldBe(Vector[4](-1, -1, 0, 0), Vector[4](1, -1, 0, 0), Vector[4](1, 1, 0, 0), Vector[4](-1, 1, 0, 0))
    faceXY.asSeq shouldBe Seq(Vector[4](-1, -1, 0, 0), Vector[4](1, -1, 0, 0), Vector[4](1, 1, 0, 0), Vector[4](-1, 1, 0, 0))

  it should "translate correctly when adding a vector" in :
    val translated = faceXY + Vector[4](2, 3, 4, 5)
    translated shouldBe Face4D(Vector[4](1, 2, 4, 5), Vector[4](3, 2, 4, 5), Vector[4](3, 4, 4, 5), Vector[4](1, 4, 4, 5))

  it should "scale correctly when dividing by a scalar" in :
    val face = Face4D(Vector[4](-2, -2, 0, 0), Vector[4](2, -2, 0, 0), Vector[4](2, 2, 0, 0), Vector[4](-2, 2, 0, 0))
    val scaled = face / 2.0f
    scaled shouldBe faceXY

  it should "test equality correctly" in :
    val face2 = Face4D(Vector[4](-1, -1, 0, 0), Vector[4](1, -1, 0, 0), Vector[4](1, 1, 0, 0), Vector[4](-1, 1, 0, 0))
    val face3 = Face4D(Vector[4](-2, -2, 0, 0), Vector[4](2, -2, 0, 0), Vector[4](2, 2, 0, 0), Vector[4](-2, 2, 0, 0))

    faceXY == face2 shouldBe true
    faceXY == face3 shouldBe false

  "plane property" should "return the correct plane for a face" in :
    faceXY.plane should be(Plane.xy)
    faceXZ.plane should be(Plane.xz)

  "4D normals" should "fail if created with only one vector" in:
    an [IllegalArgumentException] should be thrownBy normals(Seq(Vector.Z))

  it should "fail if created with three vectors" in:
    an [IllegalArgumentException] should be thrownBy normals(Seq(Vector.X, Vector.Y, Vector.Z))

  it should "fail with collinear vectors" in :
    Seq(Vector.X, Vector.Y, Vector.Z, Vector.W).foreach(vec =>
      an [IllegalArgumentException] should be thrownBy normals(Seq(vec, vec))
    )

  Seq(Vector.X, Vector.Y, Vector.Z, Vector.W).combinations(2).foreach { case Seq(vec1, vec2) =>
    it should s"be orthogonal to both ${vec1.toString} and ${vec2.toString}" in:
      forAll (normals(Seq(vec1, vec2))) { normal =>
        normal * vec1 should be (0)
        normal * vec2 should be (0)
      }
  }

  "setIndices" should "return nothing with null vector" in:
    setIndices(Vector.Zero[4]) shouldBe empty

  Seq(Vector.X, Vector.Y, Vector.Z, Vector.W).foreach { vec =>
    it should s"return 1 index with ${vec.toString}" in :
      setIndices(vec) should have length 1
  }

  Seq(Vector.X, Vector.Y, Vector.Z, Vector.W).combinations(2).foreach {
    case List[Vector[4]](v1, v2) =>
      it should s"return 2 indices with ${v1 + v2}" in :
        setIndices(v1 + v2) should have length 2
  }

  Seq(-Vector.X, -Vector.Y, -Vector.Z, -Vector.W).combinations(2).foreach {
    case List[Vector[4]] (v1, v2) =>
      it should s"return 2 indices swith ${v1 + v2}" in :
        setIndices(v1 + v2) should have length 2
  }

  Seq(Vector.X, Vector.Y, Vector.Z, Vector.W).combinations(3).foreach {
    case List[Vector[4]](v1, v2, v3) =>
      it should s"return 3 indices with ${v1 + v2 + v3}" in :
        setIndices(v1 + v2 + v3) should have length 3
  }

  Seq(-Vector.X, -Vector.Y, -Vector.Z, -Vector.W).combinations(3).foreach {
    case List[Vector[4]](v1, v2, v3) =>
      it should s"return 3 indices with ${v1 + v2 + v3}" in :
        setIndices(v1 + v2 + v3) should have length 3
  }

  List(0 -> Vector.X, 1 -> Vector.Y, 2 -> Vector.Z, 3 -> Vector.W).combinations(2).foreach {
      case List((i1, v1), (i2, v2)) =>
        it should s"return $i1 and $i2 for ${(v1 + v2).toString}" in:
          setIndices(v1 + v2) should be (Seq(i1, i2))
      case _ =>
  }

  "instantiating a Face4D from its vertices" should "create the normals" in:
    faceXY.normals.head should not be Vector.Zero[4]
    faceXY.normals.last should not be Vector.Zero[4]

  it should "create a face from a sequence of vectors" in:
    val vectors = Seq(Vector[4](-1, -1, 0, 0), Vector[4](1, -1, 0, 0), Vector[4](1, 1, 0, 0), Vector[4](-1, 1, 0, 0))
    val face = Face4D(vectors)
    face shouldBe Face4D(Vector[4](-1, -1, 0, 0), Vector[4](1, -1, 0, 0), Vector[4](1, 1, 0, 0), Vector[4](-1, 1, 0, 0))

  it should "throw exception when given less than 4 points" in:
    val vectors = Seq(Vector[4](-1, -1, 0, 0), Vector[4](1, -1, 0, 0), Vector[4](1, 1, 0, 0))
    an [IllegalArgumentException] should be thrownBy Face4D(vectors)

  it should "throw exception when vertices are not all same distance apart" in:
    val unevenVertices = Seq(Vector[4](-1, -1, 0, 0), Vector[4](2, -1, 0, 0), Vector[4](2, 1, 0, 0), Vector[4](-1, 1, 0, 0))
    an [IllegalArgumentException] should be thrownBy Face4D(unevenVertices)

  it should "throw exception when edges are not parallel to axes" in:
    val invalidPoints = Seq(Vector[4](-1, -1, 0, 0), Vector[4](1, -0.9f, 0, 0), Vector[4](1, 1, 0, 0), Vector[4](-1, 1, 0, 0))
    an [IllegalArgumentException] should be thrownBy Face4D(invalidPoints)

  it should "throw exception when edges are not orthogonal" in:
    val invalidPoints = Seq(Vector[4](-1, -1, 0, 0), Vector[4](1, -1, 0, 0), Vector[4](0, 1, 0, 0), Vector[4](-1, 1, 0, 0))
    an [IllegalArgumentException] should be thrownBy Face4D(invalidPoints)

  "signs of the normals" should "be +/+ when starting in the positive sense in both face edge directions" in:
    val firstEdges = Seq(Vector[4](1, 0, 0, 0), Vector[4](0, 1, 0, 0))
    normalSigns(firstEdges) should be (Seq(1.0, 1.0))

  it should "be -/+ when starting in the negative sense in the first edge direction" in:
    val firstEdges = Seq(Vector[4](-1, 0, 0, 0), Vector[4](0, 1, 0, 0))
    normalSigns(firstEdges) should be (Seq(-1.0, 1.0))

  it should "be +/- when starting in the negative sense in the second edge direction" in:
    val firstEdges = Seq(Vector[4](1, 0, 0, 0), Vector[4](0, -1, 0, 0))
    normalSigns(firstEdges) should be (Seq(1.0, -1.0))

  it should "be -/- when starting in the negative sense in both face edge directions" in:
    val firstEdges = Seq(Vector[4](-1, 0, 0, 0), Vector[4](0, -1, 0, 0))
    normalSigns(firstEdges) should be (Seq(-1.0, -1.0))

  it should "also work in other planes than xy" in:
    Map(
      Seq(Vector[4](1, 0, 0, 0), Vector[4](0, 1, 0, 0)) -> Seq(1.0, 1.0),
      Seq(Vector[4](-1, 0, 0, 0), Vector[4](0, 1, 0, 0)) -> Seq(-1.0, 1.0),
      Seq(Vector[4](1, 0, 0, 0), Vector[4](0, -1, 0, 0)) -> Seq(1.0, -1.0),
      Seq(Vector[4](-1, 0, 0, 0), Vector[4](0, -1, 0, 0)) -> Seq(-1.0, -1.0),
      Seq(Vector[4](1, 0, 0, 0), Vector[4](0, 0, 1, 0)) -> Seq(1.0, 1.0),
      Seq(Vector[4](-1, 0, 0, 0), Vector[4](0, 0, 1, 0)) -> Seq(-1.0, 1.0),
      Seq(Vector[4](1, 0, 0, 0), Vector[4](0, 0, -1, 0)) -> Seq(1.0, -1.0),
      Seq(Vector[4](-1, 0, 0, 0), Vector[4](0, 0, -1, 0)) -> Seq(-1.0, -1.0),
      Seq(Vector[4](1, 0, 0, 0), Vector[4](0, 0, 0, 1)) -> Seq(1.0, 1.0),
      Seq(Vector[4](-1, 0, 0, 0), Vector[4](0, 0, 0, 1)) -> Seq(-1.0, 1.0),
      Seq(Vector[4](1, 0, 0, 0), Vector[4](0, 0, 0, -1)) -> Seq(1.0, -1.0),
      Seq(Vector[4](-1, 0, 0, 0), Vector[4](0, 0, 0, -1)) -> Seq(-1.0, -1.0),
      Seq(Vector[4](0, 1, 0, 0), Vector[4](0, 0, 1, 0)) -> Seq(1.0, 1.0),
      Seq(Vector[4](0, -1, 0, 0), Vector[4](0, 0, 1, 0)) -> Seq(-1.0, 1.0),
      Seq(Vector[4](0, 1, 0, 0), Vector[4](0, 0, -1, 0)) -> Seq(1.0, -1.0),
      Seq(Vector[4](0, -1, 0, 0), Vector[4](0, 0, -1, 0)) -> Seq(-1.0, -1.0),
      Seq(Vector[4](0, 1, 0, 0), Vector[4](0, 0, 0, 1)) -> Seq(1.0, 1.0),
      Seq(Vector[4](0, -1, 0, 0), Vector[4](0, 0, 0, 1)) -> Seq(-1.0, 1.0),
      Seq(Vector[4](0, 1, 0, 0), Vector[4](0, 0, 0, -1)) -> Seq(1.0, -1.0),
      Seq(Vector[4](0, -1, 0, 0), Vector[4](0, 0, 0, -1)) -> Seq(-1.0, -1.0),
      Seq(Vector[4](0, 0, 1, 0), Vector[4](0, 0, 0, 1)) -> Seq(1.0, 1.0),
      Seq(Vector[4](0, 0, -1, 0), Vector[4](0, 0, 0, 1)) -> Seq(-1.0, 1.0),
      Seq(Vector[4](0, 0, 1, 0), Vector[4](0, 0, 0, -1)) -> Seq(1.0, -1.0),
      Seq(Vector[4](0, 0, -1, 0), Vector[4](0, 0, 0, -1)) -> Seq(-1.0, -1.0)
    ).foreach { case (edges, signs) =>
      normalSigns(edges) should be (signs)
    }

  "instantiating a Face4D from its vertices"  should "create the correct normals in xy" in:
    faceXY.normals should contain only (Vector.Z, Vector.W)

  it should "create the correct normals in xz" in:
    faceXZ.normals should contain only (Vector.Y, Vector.W)

  it should "create the correct normals in xw" in:
    faceXW.normals should contain only (Vector.Y, Vector.Z)

  it should "create the correct normals in yz" in:
    faceYZ.normals should contain only (Vector.X, Vector.W)

  it should "create the correct normals in yw" in:
    faceYW.normals should contain only (Vector.X, Vector.Z)

  it should "create the correct normals in zw" in:
    faceZW.normals should contain only (Vector.X, Vector.Y)

  it should "create the correct normals in -xy" in:
    val face = Face4D(Vector[4](1, -1, 0, 0), Vector[4](-1, -1, 0, 0), Vector[4](-1, 1, 0, 0), Vector[4](1, 1, 0, 0))
    face.normals should contain only (-Vector.Z, Vector.W)

  it should "create the correct normals in x-y" in:
    val face = Face4D(Vector[4](-1, 1, 0, 0), Vector[4](-1, -1, 0, 0), Vector[4](1, -1, 0, 0), Vector[4](1, 1, 0, 0))
    face.normals should contain only (Vector.Z, -Vector.W)

  it should "create the correct normals in -x-y" in:
    val face = Face4D(Vector[4](1, 1, 0, 0), Vector[4](-1, 1, 0, 0), Vector[4](-1, -1, 0, 0), Vector[4](1, -1, 0, 0))
    face.normals should contain only (-Vector.Z, -Vector.W)

  "taking two consecutive corners out of a face" should "return the other corners in the correct order" in:
    val cornersToRemove = Seq(Vector[4](-1, -1, 0, 0), Vector[4](1, -1, 0, 0))
    val remaining = remainingCorners(seqXY, cornersToRemove)
    remaining should contain only (Vector[4](1, 1, 0, 0), Vector[4](-1, 1, 0, 0))

  it should "work for the middle two corners" in:
    val cornersToRemove = Seq(Vector[4](1, -1, 0, 0), Vector[4](1, 1, 0, 0))
    val remaining = remainingCorners(seqXY, cornersToRemove)
    remaining should contain only (Vector[4](-1, 1, 0, 0), Vector[4](-1, -1, 0, 0))

  it should "work for the last two corners" in:
    val cornersToRemove = Seq(Vector[4](1, 1, 0, 0), Vector[4](-1, 1, 0, 0))
    val remaining = remainingCorners(seqXY, cornersToRemove)
    remaining should contain only (Vector[4](-1, -1, 0, 0), Vector[4](1, -1, 0, 0))

  it should "work for the last and first corner" in:
    val cornersToRemove = Seq(Vector[4](-1, 1, 0, 0), Vector[4](-1, -1, 0, 0))
    val remaining = remainingCorners(seqXY, cornersToRemove)
    remaining should contain only (Vector[4](1, -1, 0, 0), Vector[4](1, 1, 0, 0))

  it should "fail if the number of corners is wrong" in:
    val seq = seqXY.take(3)
    val cornersToRemove = Seq(Vector[4](-1, 1, 0, 0), Vector[4](-1, -1, 0, 0))
    an [IllegalArgumentException] should be thrownBy remainingCorners(seq, cornersToRemove)

  it should "fail if the number of corners to remove is wrong" in:
    val cornersToRemove = Seq(Vector[4](-1, 1, 0, 0))
    an [IllegalArgumentException] should be thrownBy remainingCorners(seqXY, cornersToRemove)

  it should "fail if the corners are not adjacent" in:
    val cornersToRemove = Seq(Vector[4](1, -1, 0, 0), Vector[4](-1, 1, 0, 0))
    an [IllegalArgumentException] should be thrownBy remainingCorners(seqXY, cornersToRemove)

  it should "fail if any of the corners to remove is not part of the face" in:
    val cornersToRemove = Seq(Vector[4](-2, -2, 0, 0), Vector[4](1, -1, 0, 0))
    an [IllegalArgumentException] should be thrownBy remainingCorners(seqXY, cornersToRemove)

  "rotate around an edge" should "return a Face4D at all" in:
    val rotated = faceXY.rotate(Edge(Vector[4](-1, -1, 0, 0), Vector[4](1, -1, 0, 0)))
    rotated should not be empty

  it should "actually return 2 Face4Ds" in:
    val rotated = faceXY.rotate(Edge(Vector[4](-1, -1, 0, 0), Vector[4](1, -1, 0, 0)))
    rotated should have length 2

  it should "rotate a selected Face4D correctly" in:
    val face = faceXY + Vector[4](0, 0, -3, -3)
    face.normals should contain (Vector.Z)
    val rotated = face.rotate(Edge(Vector[4](-1, -1, -3, -3), Vector[4](1, -1, -3, -3)))
    rotated should contain (Face4D(Vector[4](-1, -1, -3, -3), Vector[4](1, -1, -3, -3), Vector[4](1, -1, -1, -3), Vector[4](-1, -1, -1, -3)))

  it should "rotate another selected Face4D correctly" in:
    val face = Face4D(Vector[4](1, 1, 3, 3), Vector[4](-1, 1, 3, 3), Vector[4](-1, -1, 3, 3), Vector[4](1, -1, 3, 3))
    face.normals should contain (-Vector.Z)
    val rotated = face.rotate(Edge(Vector[4](1, 1, 3, 3), Vector[4](-1, 1, 3, 3)))
    rotated should contain (Face4D(Vector[4](1, 1, 3, 3), Vector[4](-1, 1, 3, 3), Vector[4](-1, 1, 1, 3), Vector[4](1, 1, 1, 3)))

  it should "throw exception when corners are not in the face" in:
    val face = Face4D(Vector[4](-1, -1, 0, 0), Vector[4](1, -1, 0, 0), Vector[4](1, 1, 0, 0), Vector[4](-1, 1, 0, 0))
    an [IllegalArgumentException] should be thrownBy face.rotate(Edge(Vector[4](-2, -2, 0, 0), Vector[4](2, -2, 0, 0)))

  "rotate" should "return 8 Face4Ds" in:
    val rotated = faceXY.rotate()
    rotated should have length 8

  it should "cover all edges of the original Face4D" in:
    val rotated = faceXY.rotate()
    val rotatedEdges = rotated.flatMap(_.edges)
    rotatedEdges should contain allElementsOf faceXY.edges

  it should "all have the same area as the original" in:
    val rotated = faceXY.rotate()
    rotated.map(_.area) should contain only faceXY.area
