package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import com.badlogic.gdx.math.Vector4
import org.scalatest.Tag

class Face4DSuite extends AnyFlatSpec with RectMesh with Matchers:

  "4D normals" should "fail if created with only one vector" in:
    assertThrows[IllegalArgumentException](normals(Seq(Vector4.Z)))

  it should "fail if created with three vectors" in:
    assertThrows[IllegalArgumentException](normals(Seq(Vector4.X, Vector4.Y, Vector4.Z)))

  it should "fail with collinear vectors" in :
    Seq(Vector4.X, Vector4.Y, Vector4.Z, Vector4.W).foreach(vec =>
      assertThrows[IllegalArgumentException](normals(Seq(vec, vec)))
    )

  Seq(Vector4.X, Vector4.Y, Vector4.Z, Vector4.W).combinations(2).foreach { case Seq(vec1, vec2) =>
    it should s"be orthogonal to both ${vec2string(vec1)} and ${vec2string(vec2)}" in:
      val n = normals(Seq(vec1, vec2))
      assert(n.forall(_.dot(vec1) == 0))
      assert(n.forall(_.dot(vec2) == 0))
  }

  "setIndices" should "return nothing with null vector" in:
    assert(setIndices(Vector4.Zero).isEmpty)

  Seq(Vector4.X, Vector4.Y, Vector4.Z, Vector4.W).foreach { vec =>
    it should s"return 1 index with ${vec2string(vec)}" in :
      val indices = setIndices(vec)
      assert(indices.length == 1)
  }

  Seq(Vector4.X, Vector4.Y, Vector4.Z, Vector4.W).combinations(2).foreach {
    case List[Vector4](v1, v2) =>
      it should s"return 2 indices with ${vec2string(v1 + v2)}" in :
        val indices = setIndices(v1 + v2)
        assert(indices.length == 2)
  }

  Seq(-Vector4.X, -Vector4.Y, -Vector4.Z, -Vector4.W).combinations(2).foreach {
    case List[Vector4
    ] (v1, v2) =>
      it should s"return 2 indices swith ${vec2string(v1 + v2)}" in :
        val indices = setIndices(v1 + v2)
        assert(indices.length == 2)
  }

  Seq(Vector4.X, Vector4.Y, Vector4.Z, Vector4.W).combinations(3).foreach {
    case List[Vector4](v1, v2, v3) =>
      it should s"return 3 indices with ${vec2string(v1 + v2 + v3)}" in :
        val indices = setIndices(v1 + v2 + v3)
        assert(indices.length == 3)
  }

  Seq(-Vector4.X, -Vector4.Y, -Vector4.Z, -Vector4.W).combinations(3).foreach {
    case List[Vector4](v1, v2, v3) =>
      it should s"return 3 indices with ${vec2string(v1 + v2 + v3)}" in :
        val indices = setIndices(v1 + v2 + v3)
        assert(indices.length == 3)
  }

  List(0 -> Vector4.X, 1 -> Vector4.Y, 2 -> Vector4.Z, 3 -> Vector4.W).combinations(2).foreach {
      case List((i1, v1), (i2, v2)) =>
        it should s"return $i1 and $i2 for ${vec2string(v1 + v2)}" in:
          val indices = setIndices(v1 + v2)
          assert(indices == Seq(i1, i2))
      case _ =>
  }

  "instantiating a Face4D from its vertices" should "create the normals" in:
    val face = Face4D(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))
    assert(face.normals.head != Vector4.Zero)
    assert(face.normals.last != Vector4.Zero)

  "signs of the normals" should "be +/+ when starting in the positive sense in both face edge directions" in:
    val firstEdges = Seq(Vector4(1, 0, 0, 0), Vector4(0, 1, 0, 0))
    assert(normalSigns(firstEdges) == Seq(1.0, 1.0))

  it should "be -/+ when starting in the negative sense in the first edge direction" in:
    val firstEdges = Seq(Vector4(-1, 0, 0, 0), Vector4(0, 1, 0, 0))
    assert(normalSigns(firstEdges) == Seq(-1.0, 1.0))

  it should "be +/- when starting in the negative sense in the second edge direction" in:
    val firstEdges = Seq(Vector4(1, 0, 0, 0), Vector4(0, -1, 0, 0))
    assert(normalSigns(firstEdges) == Seq(1.0, -1.0))

  it should "be -/- when starting in the negative sense in both face edge directions" in:
    val firstEdges = Seq(Vector4(-1, 0, 0, 0), Vector4(0, -1, 0, 0))
    assert(normalSigns(firstEdges) == Seq(-1.0, -1.0))

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
      assert(normalSigns(edges) == signs)
    }

  "instantiating a Face4D from its vertices"  should "create the correct normals in xy" in:
    val face = Face4D(Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0))
    assert(face.normals.toSet == Set (Vector4.Z, Vector4.W))

  it should "create the correct normals in xz" in:
    val face = Face4D(Vector4(-1, 0, -1, 0), Vector4(1, 0, -1, 0), Vector4(1, 0, 1, 0), Vector4(-1, 0, 1, 0))
    assert(face.normals.toSet == Set (Vector4.Y, Vector4.W))

  it should "create the correct normals in xw" in:
    val face = Face4D(Vector4(-1, 0, 0, -1), Vector4(1, 0, 0, -1), Vector4(1, 0, 0, 1), Vector4(-1, 0, 0, 1))
    assert(face.normals.toSet == Set (Vector4.Y, Vector4.Z))

  it should "create the correct normals in yz" in:
    val face = Face4D(Vector4(0, -1, -1, 0), Vector4(0, 1, -1, 0), Vector4(0, 1, 1, 0), Vector4(0, -1, 1, 0))
    assert(face.normals.toSet == Set(Vector4.X, Vector4.W))

  it should "create the correct normals in yw" in:
    val face = Face4D(Vector4(0, -1, 0, -1), Vector4(0, 1, 0, -1), Vector4(0, 1, 0, 1), Vector4(0, -1, 0, 1))
    assert(face.normals.toSet == Set(Vector4.X, Vector4.Z))

  it should "create the correct normals in zw" in:
    val face = Face4D(Vector4(0, 0, -1, -1), Vector4(0, 0, 1, -1), Vector4(0, 0, 1, 1), Vector4(0, 0, -1, 1))
    assert(face.normals.toSet == Set(Vector4.X, Vector4.Y))

  it should "create the correct normals in -xy" in:
    val face = Face4D(Vector4(1, -1, 0, 0), Vector4(-1, -1, 0, 0), Vector4(-1, 1, 0, 0), Vector4(1, 1, 0, 0))
    assert(face.normals.toSet == Set(-Vector4.Z, Vector4.W))  // TODO might need to swap with the next case

  it should "create the correct normals in x-y" in:
    val face = Face4D(Vector4(-1, 1, 0, 0), Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0))
    assert(face.normals.toSet == Set(Vector4.Z, -Vector4.W))  // TODO might need to swap with the previous case

  it should "create the correct normals in -x-y" in:
    val face = Face4D(Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0), Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0))
    assert(face.normals.toSet == Set(-Vector4.Z, -Vector4.W))
