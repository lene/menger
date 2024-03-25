package menger.objects

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import menger.objects.Direction.{X, Y, Z}

class FaceSuite extends AnyFlatSpec:
  "any ol' Face" should "instantiate" in:
    val face = Face(0, 0, 0, 1, X)
    assert(face != null)

  "subdividing one face" should "give a total of 12 subfaces" in:
    for normal <- Seq(X, Y, Z, -X, -Y, -Z) do
      val face = Face(0, 0, 0, 1, normal)
      assert(face.subdivide().size == 12)

  "subdivided subfaces" should "have correct size" in:
    val face = Face(0, 0, 0, 1, Z)
    assert(face.subdivide().forall(_.scale == 1f/3f))

  it should "have correct size for different scales" in:
    for scale <- Seq(1f, 2f, 1/3f, 1/9f, 0.5f, 1e9f, 1e-9f) do
      val face = Face(0, 0, 0, scale, Z)
      assert(face.subdivide().forall(_.scale == scale/3f))

  "8 subfaces" should "have same normals as original for all normals" in:
    for normal <- Seq(X, Y, Z, -X, -Y, -Z) do
      val face = Face(0, 0, 0, 1, normal)
      assert(face.subdivide().count(_.normal == normal) == 8)


  "unrotated subdivided subfaces" should "have correct z positions for z and -z normal" in:
    for normal <- Seq(Z, -Z) do
      val face = Face(0, 0, 0, 1, normal)
      val subfaces = face.subdivide().filter(_.normal == normal)
      assert(
        subfaces.forall(_.zCen == 0),
        s"zCen: ${subfaces.map(_.zCen)}: != 0"
      )

  it should "have correct x positions for x and -x normal" in:
    for normal <- Seq(X, -X) do
      val face = Face(0, 0, 0, 1, normal)
      val subfaces = face.subdivide().filter(_.normal == normal)
      assert(
        subfaces.forall(_.xCen == 0),
        s"xCen: ${subfaces.map(_.xCen)}: != 0"
      )

  it should "have correct y positions for y and -y normal" in:
    for normal <- Seq(Y, -Y) do
      val face = Face(0, 0, 0, 1, normal)
      val subfaces = face.subdivide().filter(_.normal == normal)
      assert(
        subfaces.forall(_.yCen == 0),
        s"xCen: ${subfaces.map(_.yCen)}: != 0"
      )

  "4 rotated subfaces" should "have rotated normals for all normals" in:
    for normal <- Seq(X, Y, Z, -X, -Y, -Z) do
      val face = Face(0, 0, 0, 1, normal)
      val subfaces = face.subdivide().filter(face => !Set(normal, -normal).contains(face.normal))
      assert(subfaces.size == 4)

  "rotated subfaces" should "be one of each required rotation for all normals" in:
    for normal <- Seq(X, Y, Z, -X, -Y, -Z) do
      val face = Face(0, 0, 0, 1, normal)
      val subfaces = face.subdivide()
      for checkedNormal <- Set(X, Y, Z, -X, -Y, -Z) -- Set(normal, -normal) do
        assert(subfaces.count(_.normal == checkedNormal) == 1)

  it should "point in correct direction for each location for x normal" in:
    val face = Face(0, 0, 0, 3, X)
    val subfaces = face.subdivide()
    subfaces.filter(_.normal == Y).foreach { f => assert(f.yCen == -0.5) }
    subfaces.filter(_.normal == -Y).foreach { f => assert(f.yCen == 0.5) }
    subfaces.filter(_.normal == Z).foreach { f => assert(f.zCen == -0.5) }
    subfaces.filter(_.normal == -Z).foreach { f => assert(f.zCen == 0.5) }

  it should "point in correct direction for each location for -x normal" in:
    val face = Face(0, 0, 0, 3, -X)
    val subfaces = face.subdivide()
    subfaces.filter(_.normal == Y).foreach { f => assert(f.yCen == 0.5) }
    subfaces.filter(_.normal == -Y).foreach { f => assert(f.yCen == -0.5) }
    subfaces.filter(_.normal == Z).foreach { f => assert(f.zCen == 0.5) }
    subfaces.filter(_.normal == -Z).foreach { f => assert(f.zCen == -0.5) }

  it should "point in correct direction for each location for y normal" in:
    val face = Face(0, 0, 0, 3, Y)
    val subfaces = face.subdivide()
    subfaces.filter(_.normal == X).foreach { f => assert(f.xCen == -0.5, s"$f") }
    subfaces.filter(_.normal == -X).foreach { f => assert(f.xCen == 0.5) }
    subfaces.filter(_.normal == Z).foreach { f => assert(f.zCen == -0.5) }
    subfaces.filter(_.normal == -Z).foreach { f => assert(f.zCen == 0.5) }

  it should "point in correct direction for each location for -y normal" in:
    val face = Face(0, 0, 0, 3, -Y)
    val subfaces = face.subdivide()
    subfaces.filter(_.normal == X).foreach { f => assert(f.xCen == 0.5, s"$f") }
    subfaces.filter(_.normal == -X).foreach { f => assert(f.xCen == -0.5) }
    subfaces.filter(_.normal == Z).foreach { f => assert(f.zCen == 0.5) }
    subfaces.filter(_.normal == -Z).foreach { f => assert(f.zCen == -0.5) }

  it should "point in correct direction for each location for z normal" in:
    val face = Face(0, 0, 0, 3, Z)
    val subfaces = face.subdivide()
    subfaces.filter(_.normal == Y).foreach { f => assert(f.yCen == -0.5) }
    subfaces.filter(_.normal == -Y).foreach { f => assert(f.yCen == 0.5) }
    subfaces.filter(_.normal == X).foreach { f => assert(f.xCen == -0.5) }
    subfaces.filter(_.normal == -X).foreach { f => assert(f.xCen == 0.5) }

  it should "point in correct direction for each location for -z normal" in:
    val face = Face(0, 0, 0, 3, -Z)
    val subfaces = face.subdivide()
    subfaces.filter(_.normal == Y).foreach { f => assert(f.yCen == 0.5) }
    subfaces.filter(_.normal == -Y).foreach { f => assert(f.yCen == -0.5) }
    subfaces.filter(_.normal == X).foreach { f => assert(f.xCen == 0.5) }
    subfaces.filter(_.normal == -X).foreach { f => assert(f.xCen == -0.5) }

  it should "have correct x positions for x normal" in:
    val face = Face(0, 0, 0, 1, X)
    val subfaces = face.subdivide().filter(_.normal != X)
    assert(
      subfaces.forall(_.xCen == -1 / 6f),
      s"xCen: ${subfaces.map(_.xCen)}: mismatch with ${-1 / 6f}"
    )

  it should "have correct x positions for -x normal" in:
    val face = Face(0, 0, 0, 1, -X)
    val subfaces = face.subdivide().filter(_.normal != -X)
    assert(
      subfaces.forall(_.xCen == 1 / 6f),
      s"xCen: ${subfaces.map(_.xCen)}: mismatch with ${1 / 6f}"
    )

  it should "have correct y positions for y normal" in:
    val face = Face(0, 0, 0, 1, Y)
    val subfaces = face.subdivide().filter(_.normal != Y)
    assert(
      subfaces.forall(_.yCen == -1 / 6f),
      s"yCen: ${subfaces.map(_.yCen)}: mismatch with ${-1 / 6f}"
    )

  it should "have correct y positions for -y normal" in:
    val face = Face(0, 0, 0, 1, -Y)
    val subfaces = face.subdivide().filter(_.normal != -Y)
    assert(
      subfaces.forall(_.yCen == 1 / 6f),
      s"yCen: ${subfaces.map(_.yCen)}: mismatch with ${1 / 6f}"
    )

  it should "have correct z positions for z normal" in:
    val face = Face(0, 0, 0, 1, Z)
    val subfaces = face.subdivide().filter(_.normal != Z)
    assert(
      subfaces.forall(_.zCen == -1 / 6f),
      s"zCen: ${subfaces.map(_.zCen)}: mismatch with ${-1 / 6f}"
    )

  it should "have correct z positions for -z normal" in:
    val face = Face(0, 0, 0, 1, -Z)
    val subfaces = face.subdivide().filter(_.normal != -Z)
    assert(
      subfaces.forall(_.zCen == 1 / 6f),
      s"zCen: ${subfaces.map(_.zCen)}: mismatch with ${1 / 6f}"
    )

  it should "have correct x positions for different scales" in:
    Seq(X, -X).foreach { normal =>
      Seq(1f, 2f, 1 / 3f, 1 / 9f, 0.5f, 1e9f, 1e-9f).foreach { scale =>
        val face = Face(0, 0, 0, scale, normal)
        val subfaces = face.subdivide().filter(_.normal != normal)
        assert(
          subfaces.forall(_.xCen == -normal.sign * scale / 6f),
          s"xCen: ${subfaces.map(_.xCen)}: mismatch with ${-normal.sign * scale / 6f}")
      }
    }

  it should "have correct y positions for different scales" in:
    Seq(Y, -Y).foreach { normal =>
      Seq(1f, 2f, 1 / 3f, 1 / 9f, 0.5f, 1e9f, 1e-9f).foreach { scale =>
        val face = Face(0, 0, 0, scale, normal)
        val subfaces = face.subdivide().filter(_.normal != normal)
        assert(
          subfaces.forall(_.yCen == -normal.sign * scale / 6f),
          s"yCen: ${subfaces.map(_.yCen)}: mismatch with ${-normal.sign * scale / 6f}")
      }
    }

  it should "have correct z positions for different scales" in:
    Seq(Z, -Z).foreach { normal =>
      Seq(1f, 2f, 1 / 3f, 1 / 9f, 0.5f, 1e9f, 1e-9f).foreach { scale =>
        val face = Face(0, 0, 0, scale, normal)
        val subfaces = face.subdivide().filter(_.normal != normal)
        assert(
          subfaces.forall(_.zCen == -normal.sign * scale / 6f),
          s"zCen: ${subfaces.map(_.zCen)}: mismatch with ${-normal.sign * scale / 6f}")
      }
    }

  "subfaces of face with non-zero center" should "keep center equal in normal direction" in:
    val face = Face(0, 0, 1, 1, Z)
    val subfaces = face.subdivide()
    assert(
      subfaces.filter(_.normal == Z).forall(_.zCen == 1),
      s"zCen: ${subfaces.map(_.zCen)}: mismatch with 1"
    )

  "subdividing any face twice" should "give a total of 12*12 subfaces" in:
    for normal <- Seq(X, Y, Z, -X, -Y, -Z) do
      val face = Face(0, 0, 0, 1, normal)
      val twiceSubdivided = face.subdivide().flatMap(_.subdivide())
      assert(twiceSubdivided.size == 144)

  it should "end up with normals pointing in all directions" in:
    for normal <- Seq(X, Y, Z, -X, -Y, -Z) do
      val face = Face(0, 0, 0, 1, normal)
      val twiceSubdivided = face.subdivide().flatMap(_.subdivide())
      assert(twiceSubdivided.map(_.normal).toSet == Set(X, Y, Z, -X, -Y, -Z))

  it should "end up with subfaces 1/9 the original size" in:
    for normal <- Seq(X, Y, Z, -X, -Y, -Z) do
      val face = Face(0, 0, 0, 1, normal)
      val twiceSubdivided = face.subdivide().flatMap(_.subdivide())
      assert(twiceSubdivided.forall(_.scale == 1f/9f))

  "a face's vertices" should "be 4" in:
    for normal <- Seq(X, Y, Z, -X, -Y, -Z) do
      val vertices = Face(0, 0, 0, 1, normal).vertices
      assert(vertices.size == 4)

  it should "be correct for +/-x normal" in:
    for normal <- Seq(X, -X) do
      val vertices = Face(0, 0, 0, 1, normal).vertices
      assert(vertices.toList.map(_.position).toSet == Set(
        Vector3(0, -0.5f, -0.5f), Vector3(0, 0.5f, -0.5f),
        Vector3(0, 0.5f, 0.5f), Vector3(0, -0.5f, 0.5f)
      ))

  it should "be correct for +/-y normal" in:
    for normal <- Seq(Y, -Y) do
      val vertices = Face(0, 0, 0, 1, normal).vertices
      assert(vertices.toList.map(_.position).toSet == Set(
        Vector3(-0.5f, 0, -0.5f), Vector3(0.5f, 0, -0.5f),
        Vector3(0.5f, 0, 0.5f), Vector3(-0.5f, 0, 0.5f)
      ))

  it should "be correct for +/-z normal" in:
    for normal <- Seq(Z, -Z) do
      val vertices = Face(0, 0, 0, 1, normal).vertices
      assert(vertices.toList.map(_.position).toSet == Set(
        Vector3(-0.5f, -0.5f, 0), Vector3(0.5f, -0.5f, 0),
        Vector3(0.5f, 0.5f, 0), Vector3(-0.5f, 0.5f, 0)
      ))

  "vertices of a face" should "be correct for different sizes" in:
    for size <- Seq(1f, 2f, 1e9f, 0.1f, 1e-9f) do
      val vertices = Face(0, 0, 0, size, X).vertices
      val half = size / 2
      assert(vertices.toList.map(_.position).toSet == Set(
        Vector3(0, -half, -half), Vector3(0, half, -half),
        Vector3(0, half, half), Vector3(0, -half, half)
      ))

  it should "be correct for different centers" in:
    val half = 0.5f
    for (xCen, yCen, zCen) <- Seq(
      (0f, 0f, 1f), (0f, 1f, 0f), (1f, 0f, 0f), (0f, 0f, -1f), (0f, -1f, 0f), (-1f, 0f, 0f),
      (1f, 1f, 1f), (1f, 1f, -1f), (1f, -1f, 1f), (1f, -1f, -1f),
      (-1f, 1f, 1f), (-1f, 1f, -1f), (-1f, -1f, 1f), (-1f, -1f, -1f)
    ) do
      val vertices = Face(xCen, yCen, zCen, 1, X).vertices
      assert(vertices.toList.map(_.position).toSet == Set(
        Vector3(xCen, yCen - half, zCen - half), Vector3(xCen, yCen + half, zCen - half),
        Vector3(xCen, yCen + half, zCen + half), Vector3(xCen, yCen - half, zCen + half)
      ))
