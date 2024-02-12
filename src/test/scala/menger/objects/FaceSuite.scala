package menger.objects

import org.scalatest.funsuite.AnyFunSuite
import menger.objects.Direction.{X, Y, Z}

class FaceSuite extends AnyFunSuite:
  test("instantiate any ol' Face") {
    val face = Face(0, 0, 0, 1, X)
    assert(face != null)
  }

  test("subdividing one face gives a total of 12 subfaces") {
    for normal <- Seq(X, Y, Z, -X, -Y, -Z) do
      val face = Face(0, 0, 0, 1, normal)
      assert(face.subdivide().size == 12)
  }

  test("subdivided subfaces have correct size") {
    val face = Face(0, 0, 0, 1, Z)
    assert(face.subdivide().forall(_.scale == 1f/3f))
  }

  test("subdivided subfaces have correct size for different scales") {
    Seq(1f, 2f, 1/3f, 1/9f, 0.5f, 1e9f, 1e-9f).foreach { scale =>
      val face = Face(0, 0, 0, scale, Z)
      assert(face.subdivide().forall(_.scale == scale/3f))
    }
  }

  test("8 subfaces have same normals as original for all normals") {
    for normal <- Seq(X, Y, Z, -X, -Y, -Z) do
      val face = Face(0, 0, 0, 1, normal)
      assert(face.subdivide().count(_.normal == normal) == 8)
  }

  test("unrotated subdivided subfaces have correct z positions for z and -z normal") {
    for normal <- Seq(Z, -Z) do
      val face = Face(0, 0, 0, 1, normal)
      val subfaces = face.subdivide().filter(_.normal == normal)
      assert(
        subfaces.forall(_.zCen == 0),
        s"zCen: ${subfaces.map(_.zCen)}: != 0"
      )
  }

  test("unrotated subdivided subfaces have correct x positions for x and -x normal") {
    for normal <- Seq(X, -X) do
      val face = Face(0, 0, 0, 1, normal)
      val subfaces = face.subdivide().filter(_.normal == normal)
      assert(
        subfaces.forall(_.xCen == 0),
        s"xCen: ${subfaces.map(_.xCen)}: != 0"
      )
  }

  test("unrotated subdivided subfaces have correct y positions for y and -y normal") {
    for normal <- Seq(Y, -Y) do
      val face = Face(0, 0, 0, 1, normal)
      val subfaces = face.subdivide().filter(_.normal == normal)
      assert(
        subfaces.forall(_.yCen == 0),
        s"xCen: ${subfaces.map(_.yCen)}: != 0"
      )
  }

  test("4 rotated subfaces have rotated normals for all normals") {
    for normal <- Seq(X, Y, Z, -X, -Y, -Z) do
      val face = Face(0, 0, 0, 1, normal)
      val subfaces = face.subdivide().filter(face => !Set(normal, -normal).contains(face.normal))
      assert(subfaces.size == 4)
  }

  test("rotated subfaces are one of each required rotation for all normals") {
    for normal <- Seq(X, Y, Z, -X, -Y, -Z) do
      val face = Face(0, 0, 0, 1, normal)
      val subfaces = face.subdivide()
      for checkedNormal <- Set(X, Y, Z, -X, -Y, -Z) -- Set(normal, -normal) do
        assert(subfaces.count(_.normal == checkedNormal) == 1)
  }

  test("rotated subfaces point in correct direction for each location for x normal") {
    val face = Face(0, 0, 0, 3, X)
    val subfaces = face.subdivide()
    subfaces.filter(_.normal == Y).foreach { f => assert(f.yCen == -1) }
    subfaces.filter(_.normal == -Y).foreach { f => assert(f.yCen == 1) }
    subfaces.filter(_.normal == Z).foreach { f => assert(f.zCen == -1) }
    subfaces.filter(_.normal == -Z).foreach { f => assert(f.zCen == 1) }
  }

  test("rotated subfaces point in correct direction for each location for -x normal") {
    val face = Face(0, 0, 0, 3, -X)
    val subfaces = face.subdivide()
    subfaces.filter(_.normal == Y).foreach { f => assert(f.yCen == 1) }
    subfaces.filter(_.normal == -Y).foreach { f => assert(f.yCen == -1) }
    subfaces.filter(_.normal == Z).foreach { f => assert(f.zCen == 1) }
    subfaces.filter(_.normal == -Z).foreach { f => assert(f.zCen == -1) }
  }

  test("rotated subfaces point in correct direction for each location for y normal") {
    val face = Face(0, 0, 0, 3, Y)
    val subfaces = face.subdivide()
    subfaces.filter(_.normal == X).foreach { f => assert(f.xCen == -1, s"$f") }
    subfaces.filter(_.normal == -X).foreach { f => assert(f.xCen == 1) }
    subfaces.filter(_.normal == Z).foreach { f => assert(f.zCen == -1) }
    subfaces.filter(_.normal == -Z).foreach { f => assert(f.zCen == 1) }
  }

  test("rotated subfaces point in correct direction for each location for -y normal") {
    val face = Face(0, 0, 0, 3, -Y)
    val subfaces = face.subdivide()
    subfaces.filter(_.normal == X).foreach { f => assert(f.xCen == 1, s"$f") }
    subfaces.filter(_.normal == -X).foreach { f => assert(f.xCen == -1) }
    subfaces.filter(_.normal == Z).foreach { f => assert(f.zCen == 1) }
    subfaces.filter(_.normal == -Z).foreach { f => assert(f.zCen == -1) }
  }

  test("rotated subfaces point in correct direction for each location for z normal") {
    val face = Face(0, 0, 0, 3, Z)
    val subfaces = face.subdivide()
    subfaces.filter(_.normal == Y).foreach { f => assert(f.yCen == -1) }
    subfaces.filter(_.normal == -Y).foreach { f => assert(f.yCen == 1) }
    subfaces.filter(_.normal == X).foreach { f => assert(f.xCen == -1) }
    subfaces.filter(_.normal == -X).foreach { f => assert(f.xCen == 1) }
  }

  test("rotated subfaces point in correct direction for each location for -z normal") {
    val face = Face(0, 0, 0, 3, -Z)
    val subfaces = face.subdivide()
    subfaces.filter(_.normal == Y).foreach { f => assert(f.yCen == 1) }
    subfaces.filter(_.normal == -Y).foreach { f => assert(f.yCen == -1) }
    subfaces.filter(_.normal == X).foreach { f => assert(f.xCen == 1) }
    subfaces.filter(_.normal == -X).foreach { f => assert(f.xCen == -1) }
  }

  test("rotated subfaces have correct x positions for x normal") {
    val face = Face(0, 0, 0, 1, X)
    val subfaces = face.subdivide().filter(_.normal != X)
    assert(
      subfaces.forall(_.xCen == -1 / 3f),
      s"xCen: ${subfaces.map(_.xCen)}: mismatch with ${-1 / 3f}"
    )
  }

  test("rotated subfaces have correct x positions for -x normal") {
    val face = Face(0, 0, 0, 1, -X)
    val subfaces = face.subdivide().filter(_.normal != -X)
    assert(
      subfaces.forall(_.xCen == 1 / 3f),
      s"xCen: ${subfaces.map(_.xCen)}: mismatch with ${1 / 3f}"
    )
  }

  test("rotated subfaces have correct y positions for y normal") {
    val face = Face(0, 0, 0, 1, Y)
    val subfaces = face.subdivide().filter(_.normal != Y)
    assert(
      subfaces.forall(_.yCen == -1 / 3f),
      s"yCen: ${subfaces.map(_.yCen)}: mismatch with ${-1 / 3f}"
    )
  }

  test("rotated subfaces have correct y positions for -y normal") {
    val face = Face(0, 0, 0, 1, -Y)
    val subfaces = face.subdivide().filter(_.normal != -Y)
    assert(
      subfaces.forall(_.yCen == 1 / 3f),
      s"yCen: ${subfaces.map(_.yCen)}: mismatch with ${1 / 3f}"
    )
  }

  test("rotated subfaces have correct z positions for z normal") {
    val face = Face(0, 0, 0, 1, Z)
    val subfaces = face.subdivide().filter(_.normal != Z)
    assert(
      subfaces.forall(_.zCen == -1 / 3f),
      s"zCen: ${subfaces.map(_.zCen)}: mismatch with ${-1 / 3f}"
    )
  }

  test("rotated subfaces have correct z positions for -z normal") {
    val face = Face(0, 0, 0, 1, -Z)
    val subfaces = face.subdivide().filter(_.normal != -Z)
    assert(
      subfaces.forall(_.zCen == 1 / 3f),
      s"zCen: ${subfaces.map(_.zCen)}: mismatch with ${1 / 3f}"
    )
  }

  test("rotated subfaces have correct x positions for different scales") {
    Seq(X, -X).foreach { normal =>
      Seq(1f, 2f, 1 / 3f, 1 / 9f, 0.5f, 1e9f, 1e-9f).foreach { scale =>
        val face = Face(0, 0, 0, scale, normal)
        val subfaces = face.subdivide().filter(_.normal != normal)
        assert(
          subfaces.forall(_.xCen == -normal.sign * scale / 3f),
          s"xCen: ${subfaces.map(_.xCen)}: mismatch with ${-normal.sign * scale / 3f}")
      }
    }
  }

  test("rotated subfaces have correct y positions for different scales") {
    Seq(Y, -Y).foreach { normal =>
      Seq(1f, 2f, 1 / 3f, 1 / 9f, 0.5f, 1e9f, 1e-9f).foreach { scale =>
        val face = Face(0, 0, 0, scale, normal)
        val subfaces = face.subdivide().filter(_.normal != normal)
        assert(
          subfaces.forall(_.yCen == -normal.sign * scale / 3f),
          s"yCen: ${subfaces.map(_.yCen)}: mismatch with ${-normal.sign * scale / 3f}")
      }
    }
  }

  test("rotated subfaces have correct z positions for different scales") {
    Seq(Z, -Z).foreach { normal =>
      Seq(1f, 2f, 1 / 3f, 1 / 9f, 0.5f, 1e9f, 1e-9f).foreach { scale =>
        val face = Face(0, 0, 0, scale, normal)
        val subfaces = face.subdivide().filter(_.normal != normal)
        assert(
          subfaces.forall(_.zCen == -normal.sign * scale / 3f),
          s"zCen: ${subfaces.map(_.zCen)}: mismatch with ${-normal.sign * scale / 3f}")
      }
    }
  }

  test("subfaces of face with non-zero center") {
    val face = Face(0, 0, 1, 1, Z)
    val subfaces = face.subdivide()
    assert(
      subfaces.filter(_.normal == Z).forall(_.zCen == 1),
      s"zCen: ${subfaces.map(_.zCen)}: mismatch with 1"
    )
  }

  test("subdividing any face twice gives a total of 12*12 subfaces") {
    for normal <- Seq(X, Y, Z, -X, -Y, -Z) do
      val face = Face(0, 0, 0, 1, normal)
      val twiceSubdivided = face.subdivide().flatMap(_.subdivide())
      assert(twiceSubdivided.size == 144)
  }

  test("subdividing any face twice ends up with normals pointing in all directions") {
    for normal <- Seq(X, Y, Z, -X, -Y, -Z) do
      val face = Face(0, 0, 0, 1, normal)
      val twiceSubdivided = face.subdivide().flatMap(_.subdivide())
      assert(twiceSubdivided.map(_.normal).toSet == Set(X, Y, Z, -X, -Y, -Z))
  }

  test("subdividing any face twice ends up with subfaces 1/9 the original size") {
    for normal <- Seq(X, Y, Z, -X, -Y, -Z) do
      val face = Face(0, 0, 0, 1, normal)
      val twiceSubdivided = face.subdivide().flatMap(_.subdivide())
      assert(twiceSubdivided.forall(_.scale == 1f/9f))
  }