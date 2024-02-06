package menger.objects

import org.scalatest.funsuite.AnyFunSuite
import menger.objects.Direction.{X, Y, Z}

class FaceSuite extends AnyFunSuite:
  test("instantiate any ol' Face") {
    val face = Face(0, 0, 0, 1, X)
    assert(face != null)
  }

  test("subdividing one face gives a total of 12 subfaces") {
    val face = Face(0, 0, 0, 1, Z)
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

  test("8 subfaces have same normals as original") {
    val face = Face(0, 0, 0, 1, Z)
    assert(face.subdivide().count(_.normal == Z) == 8)
  }

  test("unrotated subdivided subfaces have correct z positions") {
    val face = Face(0, 0, 0, 1, Z)
    val subfaces = face.subdivide().filter(_.normal == Z)
    assert(
      subfaces.forall(_.zCen == 0),
      s"zCen: ${subfaces.map(_.zCen)}: mismatch with 0 ($subfaces)"
    )
  }

  test("4 rotated subfaces have rotated normals") {
    val face = Face(0, 0, 0, 1, Z)
    val subfaces = face.subdivide().filter(_.normal != Z)
    assert(subfaces.size == 4)
  }

  test("rotated subfaces are one of each required rotation") {
    val face = Face(0, 0, 0, 1, Z)
    val subfaces = face.subdivide()
    assert(subfaces.count(_.normal == X) == 1)
    assert(subfaces.count(_.normal == -X) == 1)
    assert(subfaces.count(_.normal == Y) == 1)
    assert(subfaces.count(_.normal == -Y) == 1)
  }

  test("rotated subfaces point in correct direction for each location") {
    val face = Face(0, 0, 0, 3, Z)
    val subfaces = face.subdivide()
    subfaces.filter(_.normal == Y).foreach { f => assert(f.yCen == -1) }
    subfaces.filter(_.normal == -Y).foreach { f => assert(f.yCen == 1) }
    subfaces.filter(_.normal == X).foreach { f => assert(f.xCen == -1) }
    subfaces.filter(_.normal == -X).foreach { f => assert(f.xCen == 1) }
  }

  test("rotated subfaces have correct z positions") {
    val face = Face(0, 0, 0, 1, Z)
    val subfaces = face.subdivide().filter(_.normal != Z)
    assert(
      subfaces.forall(_.zCen == -1/3f),
      s"zCen: ${subfaces.map(_.zCen)}: mismatch with ${-1/3f}"
    )
  }

  test("rotated subfaces have correct z positions for different scales") {
    Seq(1f, 2f, 1/3f, 1/9f, 0.5f, 1e9f, 1e-9f).foreach { scale =>
      val face = Face(0, 0, 0, scale, Z)
      val subfaces = face.subdivide().filter(_.normal != Z)
      assert(
        subfaces.forall(_.zCen == -scale/3f),
        s"zCen: ${subfaces.map(_.zCen)}: mismatch with ${-scale/3f}")
    }
  }

  test("string representation of subdivided face") {
    val face = Face(0, 0, 0, 3, Z)
    val subfaces = face.subdivide()
    println(subfaces.mkString("\n"))
  }

  ignore("rotate face around x axis") {
    println("TODO")
  }

  ignore("subdivided rotated face") {
    println("TODO")
  }