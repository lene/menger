package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Inspectors.forAll

class Face4DSuite extends AnyFlatSpec with RectMesh with Matchers:

  "4D normals" should "fail if created with only one vector" in:
    an [IllegalArgumentException] should be thrownBy normals(Seq(Vector4.Z))

  it should "fail if created with three vectors" in:
    an [IllegalArgumentException] should be thrownBy normals(Seq(Vector4.X, Vector4.Y, Vector4.Z))

  it should "fail with collinear vectors" in :
    Seq(Vector4.X, Vector4.Y, Vector4.Z, Vector4.W).foreach(vec =>
      an [IllegalArgumentException] should be thrownBy normals(Seq(vec, vec))
    )

  Seq(Vector4.X, Vector4.Y, Vector4.Z, Vector4.W).combinations(2).foreach { case Seq(vec1, vec2) =>
    it should s"be orthogonal to both ${vec1.asString} and ${vec2.asString}" in:
      forAll (normals(Seq(vec1, vec2))) { normal =>
        normal.dot(vec1) should be (0)
        normal.dot(vec2) should be (0)
      }
  }

  "setIndices" should "return nothing with null vector" in:
    setIndices(Vector4.Zero) shouldBe empty

  Seq(Vector4.X, Vector4.Y, Vector4.Z, Vector4.W).foreach { vec =>
    it should s"return 1 index with ${vec.asString}" in :
      setIndices(vec) should have length 1
  }

  Seq(Vector4.X, Vector4.Y, Vector4.Z, Vector4.W).combinations(2).foreach {
    case List[Vector4](v1, v2) =>
      it should s"return 2 indices with ${(v1 + v2).asString}" in :
        setIndices(v1 + v2) should have length 2
  }

  Seq(-Vector4.X, -Vector4.Y, -Vector4.Z, -Vector4.W).combinations(2).foreach {
    case List[Vector4
    ] (v1, v2) =>
      it should s"return 2 indices swith ${(v1 + v2).asString}" in :
        setIndices(v1 + v2) should have length 2
  }

  Seq(Vector4.X, Vector4.Y, Vector4.Z, Vector4.W).combinations(3).foreach {
    case List[Vector4](v1, v2, v3) =>
      it should s"return 3 indices with ${(v1 + v2 + v3).asString}" in :
        setIndices(v1 + v2 + v3) should have length 3
  }

  Seq(-Vector4.X, -Vector4.Y, -Vector4.Z, -Vector4.W).combinations(3).foreach {
    case List[Vector4](v1, v2, v3) =>
      it should s"return 3 indices with ${(v1 + v2 + v3).asString}" in :
        setIndices(v1 + v2 + v3) should have length 3
  }

  List(0 -> Vector4.X, 1 -> Vector4.Y, 2 -> Vector4.Z, 3 -> Vector4.W).combinations(2).foreach {
      case List((i1, v1), (i2, v2)) =>
        it should s"return $i1 and $i2 for ${(v1 + v2).asString}" in:
          setIndices(v1 + v2) should be (Seq(i1, i2))
      case _ =>
  }

  "instantiating a Face4D from its vertices" should "create the normals" in:
    val face = Face4D(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))
    face.normals.head should not be Vector4.Zero
    face.normals.last should not be Vector4.Zero

  "signs of the normals" should "be +/+ when starting in the positive sense in both face edge directions" in:
    val firstEdges = Seq(Vector4(1, 0, 0, 0), Vector4(0, 1, 0, 0))
    normalSigns(firstEdges) should be (Seq(1.0, 1.0))

  it should "be -/+ when starting in the negative sense in the first edge direction" in:
    val firstEdges = Seq(Vector4(-1, 0, 0, 0), Vector4(0, 1, 0, 0))
    normalSigns(firstEdges) should be (Seq(-1.0, 1.0))

  it should "be +/- when starting in the negative sense in the second edge direction" in:
    val firstEdges = Seq(Vector4(1, 0, 0, 0), Vector4(0, -1, 0, 0))
    normalSigns(firstEdges) should be (Seq(1.0, -1.0))

  it should "be -/- when starting in the negative sense in both face edge directions" in:
    val firstEdges = Seq(Vector4(-1, 0, 0, 0), Vector4(0, -1, 0, 0))
    normalSigns(firstEdges) should be (Seq(-1.0, -1.0))

  it should "also work in other planes than xy" in:
    Map(
      Seq(Vector4(1, 0, 0, 0), Vector4(0, 1, 0, 0)) -> Seq(1.0, 1.0),
      Seq(Vector4(-1, 0, 0, 0), Vector4(0, 1, 0, 0)) -> Seq(-1.0, 1.0),
      Seq(Vector4(1, 0, 0, 0), Vector4(0, -1, 0, 0)) -> Seq(1.0, -1.0),
      Seq(Vector4(-1, 0, 0, 0), Vector4(0, -1, 0, 0)) -> Seq(-1.0, -1.0),
      Seq(Vector4(1, 0, 0, 0), Vector4(0, 0, 1, 0)) -> Seq(1.0, 1.0),
      Seq(Vector4(-1, 0, 0, 0), Vector4(0, 0, 1, 0)) -> Seq(-1.0, 1.0),
      Seq(Vector4(1, 0, 0, 0), Vector4(0, 0, -1, 0)) -> Seq(1.0, -1.0),
      Seq(Vector4(-1, 0, 0, 0), Vector4(0, 0, -1, 0)) -> Seq(-1.0, -1.0),
      Seq(Vector4(1, 0, 0, 0), Vector4(0, 0, 0, 1)) -> Seq(1.0, 1.0),
      Seq(Vector4(-1, 0, 0, 0), Vector4(0, 0, 0, 1)) -> Seq(-1.0, 1.0),
      Seq(Vector4(1, 0, 0, 0), Vector4(0, 0, 0, -1)) -> Seq(1.0, -1.0),
      Seq(Vector4(-1, 0, 0, 0), Vector4(0, 0, 0, -1)) -> Seq(-1.0, -1.0),
      Seq(Vector4(0, 1, 0, 0), Vector4(0, 0, 1, 0)) -> Seq(1.0, 1.0),
      Seq(Vector4(0, -1, 0, 0), Vector4(0, 0, 1, 0)) -> Seq(-1.0, 1.0),
      Seq(Vector4(0, 1, 0, 0), Vector4(0, 0, -1, 0)) -> Seq(1.0, -1.0),
      Seq(Vector4(0, -1, 0, 0), Vector4(0, 0, -1, 0)) -> Seq(-1.0, -1.0),
      Seq(Vector4(0, 1, 0, 0), Vector4(0, 0, 0, 1)) -> Seq(1.0, 1.0),
      Seq(Vector4(0, -1, 0, 0), Vector4(0, 0, 0, 1)) -> Seq(-1.0, 1.0),
      Seq(Vector4(0, 1, 0, 0), Vector4(0, 0, 0, -1)) -> Seq(1.0, -1.0),
      Seq(Vector4(0, -1, 0, 0), Vector4(0, 0, 0, -1)) -> Seq(-1.0, -1.0),
      Seq(Vector4(0, 0, 1, 0), Vector4(0, 0, 0, 1)) -> Seq(1.0, 1.0),
      Seq(Vector4(0, 0, -1, 0), Vector4(0, 0, 0, 1)) -> Seq(-1.0, 1.0),
      Seq(Vector4(0, 0, 1, 0), Vector4(0, 0, 0, -1)) -> Seq(1.0, -1.0),
      Seq(Vector4(0, 0, -1, 0), Vector4(0, 0, 0, -1)) -> Seq(-1.0, -1.0)
    ).foreach { case (edges, signs) =>
      normalSigns(edges) should be (signs)
    }

  "instantiating a Face4D from its vertices"  should "create the correct normals in xy" in:
    val face = Face4D(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))
    face.normals should contain only (Vector4.Z, Vector4.W)

  it should "create the correct normals in xz" in:
    val face = Face4D(Vector4(-1, 0, -1, 0), Vector4(1, 0, -1, 0), Vector4(1, 0, 1, 0), Vector4(-1, 0, 1, 0))
    face.normals should contain only (Vector4.Y, Vector4.W)

  it should "create the correct normals in xw" in:
    val face = Face4D(Vector4(-1, 0, 0, -1), Vector4(1, 0, 0, -1), Vector4(1, 0, 0, 1), Vector4(-1, 0, 0, 1))
    face.normals should contain only (Vector4.Y, Vector4.Z)

  it should "create the correct normals in yz" in:
    val face = Face4D(Vector4(0, -1, -1, 0), Vector4(0, 1, -1, 0), Vector4(0, 1, 1, 0), Vector4(0, -1, 1, 0))
    face.normals should contain only (Vector4.X, Vector4.W)

  it should "create the correct normals in yw" in:
    val face = Face4D(Vector4(0, -1, 0, -1), Vector4(0, 1, 0, -1), Vector4(0, 1, 0, 1), Vector4(0, -1, 0, 1))
    face.normals should contain only (Vector4.X, Vector4.Z)

  it should "create the correct normals in zw" in:
    val face = Face4D(Vector4(0, 0, -1, -1), Vector4(0, 0, 1, -1), Vector4(0, 0, 1, 1), Vector4(0, 0, -1, 1))
    face.normals should contain only (Vector4.X, Vector4.Y)

  it should "create the correct normals in -xy" in:
    val face = Face4D(Vector4(1, -1, 0, 0), Vector4(-1, -1, 0, 0), Vector4(-1, 1, 0, 0), Vector4(1, 1, 0, 0))
    face.normals should contain only (-Vector4.Z, Vector4.W)  // TODO might need to swap with the next case

  it should "create the correct normals in x-y" in:
    val face = Face4D(Vector4(-1, 1, 0, 0), Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0))
    face.normals should contain only (Vector4.Z, -Vector4.W)  // TODO might need to swap with the previous case

  it should "create the correct normals in -x-y" in:
    val face = Face4D(Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0), Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0))
    face.normals should contain only (-Vector4.Z, -Vector4.W)

  "taking two consecutive corners out of a face" should "return the other corners in the correct order" in:
    val seq = Seq(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))
    val cornersToRemove = Seq(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0))
    val remaining = remainingCorners(seq, cornersToRemove)
    remaining should contain only (Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))

  it should "work for the middle two corners" in:
    val seq = Seq(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))
    val cornersToRemove = Seq(Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0))
    val remaining = remainingCorners(seq, cornersToRemove)
    remaining should contain only (Vector4(-1, 1, 0, 0), Vector4(-1, -1, 0, 0))

  it should "work for the last two corners" in:
    val seq = Seq(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))
    val cornersToRemove = Seq(Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))
    val remaining = remainingCorners(seq, cornersToRemove)
    remaining should contain only (Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0))

  it should "work for the last and first corner" in:
    val seq = Seq(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))
    val cornersToRemove = Seq(Vector4(-1, 1, 0, 0), Vector4(-1, -1, 0, 0))
    val remaining = remainingCorners(seq, cornersToRemove)
    remaining should contain only (Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0))

  it should "fail if the number of corners is wrong" in:
    val seq = Seq(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0))
    val cornersToRemove = Seq(Vector4(-1, 1, 0, 0), Vector4(-1, -1, 0, 0))
    an [IllegalArgumentException] should be thrownBy remainingCorners(seq, cornersToRemove)

  it should "fail if the number of corners to remove is wrong" in:
    val seq = Seq(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))
    val cornersToRemove = Seq(Vector4(-1, 1, 0, 0))
    an [IllegalArgumentException] should be thrownBy remainingCorners(seq, cornersToRemove)

  it should "fail if the corners are not adjacent" in:
    val seq = Seq(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))
    val cornersToRemove = Seq(Vector4(1, -1, 0, 0), Vector4(-1, 1, 0, 0))
    an [IllegalArgumentException] should be thrownBy remainingCorners(seq, cornersToRemove)

  it should "fail if any of the corners to remove is not part of the face" in:
    val seq = Seq(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))
    val cornersToRemove = Seq(Vector4(-2, -2, 0, 0), Vector4(1, -1, 0, 0))
    an [IllegalArgumentException] should be thrownBy remainingCorners(seq, cornersToRemove)

  "rotate around an edge" should "return a Face4D at all" in:
    val face = Face4D(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))
    val rotated = face.rotate(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0))
    rotated should not be empty

  it should "actually return 2 Face4Ds" in:
    val face = Face4D(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))
    val rotated = face.rotate(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0))
    rotated should have length 2

  it should "rotate a selected Face4D correctly" in:
    val face = Face4D(Vector4(-1, -1, -3, -3), Vector4(1, -1, -3, -3), Vector4(1, 1, -3, -3), Vector4(-1, 1, -3, -3))
    assert(face.normals.contains(Vector4.Z))
    val rotated = face.rotate(Vector4(-1, -1, -3, -3), Vector4(1, -1, -3, -3))
    rotated should contain (Face4D(Vector4(-1, -1, -3, -3), Vector4(1, -1, -3, -3), Vector4(1, -1, -1, -3), Vector4(-1, -1, -1, -3)))

  it should "rotate another selected Face4D correctly" in:
    val face = Face4D(Vector4(1, 1, 3, 3), Vector4(-1, 1, 3, 3), Vector4(-1, -1, 3, 3), Vector4(1, -1, 3, 3))
    assert(face.normals.contains(-Vector4.Z))
    val rotated = face.rotate(Vector4(1, 1, 3, 3), Vector4(-1, 1, 3, 3))
    rotated should contain (Face4D(Vector4(1, 1, 3, 3), Vector4(-1, 1, 3, 3), Vector4(-1, 1, 1, 3), Vector4(1, 1, 1, 3)))

  "rotate" should "return 8 Face4Ds" in:
    val face = Face4D(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))
    val rotated = face.rotate()
    rotated should have length 8

  it should "cover all edges of the original Face4D" in:
    val face = Face4D(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))
    val rotated = face.rotate()
    val rotatedEdges = rotated.flatMap(_.edges)
    rotatedEdges should contain allElementsOf face.edges

  it should "all have the same area as the original" in:
    val face = Face4D(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))
    val rotated = face.rotate()
    rotated.map(_.area) should contain only face.area
