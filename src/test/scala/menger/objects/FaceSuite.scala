package menger.objects

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Inspectors.forAll
import menger.objects.Direction.{X, Y, Z}

class FaceSuite extends AnyFlatSpec with Matchers:
  "any ol' Face" should "instantiate" in:
    val face = Face(0, 0, 0, 1, X)
    face should not be null

  "subdividing one face" should "give a total of 12 subfaces" in:
    forAll(Seq(X, Y, Z, -X, -Y, -Z)) { normal =>
      val face = Face(0, 0, 0, 1, normal)
      face.subdivide() should have size 12
    }

  "subdivided subfaces" should "have correct size" in:
    val face = Face(0, 0, 0, 1, Z)
    forAll (face.subdivide()) { _.scale should be (1f/3f) }

  it should "have correct size for different scales" in:
    for scale <- Seq(1f, 2f, 1/3f, 1/9f, 0.5f, 1e9f, 1e-9f) do
      val face = Face(0, 0, 0, scale, Z)
      forAll(face.subdivide()) { _.scale should be(scale / 3f) }

  "8 subfaces" should "have same normals as original for all normals" in:
    forAll(Seq(X, Y, Z, -X, -Y, -Z)) { normal =>
      val face = Face(0, 0, 0, 1, normal)
      face.subdivide().count(_.normal == normal) should be(8)
    }

  "unrotated subdivided subfaces" should "have correct z positions for z and -z normal" in:
    forAll(Seq(Z, -Z)) { normal =>
      val face = Face(0, 0, 0, 1, normal)
      val subfaces = face.subdivide().filter(_.normal == normal)
      forAll(subfaces) { _.zCen should be(0) }
    }

  it should "have correct x positions for x and -x normal" in:
    forAll(Seq(X, -X)) { normal =>
      val face = Face(0, 0, 0, 1, normal)
      val subfaces = face.subdivide().filter(_.normal == normal)
      forAll(subfaces) { _.xCen should be(0) }
    }

  it should "have correct y positions for y and -y normal" in:
    forAll(Seq(Y, -Y)) { normal =>
      val face = Face(0, 0, 0, 1, normal)
      val subfaces = face.subdivide().filter(_.normal == normal)
      forAll(subfaces) { _.yCen should be(0) }
    }

  "4 rotated subfaces" should "have rotated normals for all normals" in:
    forAll(Seq(X, Y, Z, -X, -Y, -Z)) { normal =>
      val face = Face(0, 0, 0, 1, normal)
      val subfaces = face.subdivide().filter(face => !Set(normal, -normal).contains(face.normal))
      subfaces should have size 4
    }

  "rotated subfaces" should "be one of each required rotation for all normals" in:
    forAll(Seq(X, Y, Z, -X, -Y, -Z)) { normal =>
      val face = Face(0, 0, 0, 1, normal)
      val subfaces = face.subdivide()
      forAll(Set(X, Y, Z, -X, -Y, -Z) -- Set(normal, -normal)) { checkedNormal =>
        subfaces.count(_.normal == checkedNormal) should be(1)
      }
    }

  it should "point in correct direction for each location for x normal" in:
    val face = Face(0, 0, 0, 3, X)
    val subfaces = face.subdivide()
    forAll(subfaces.filter(_.normal == Y)) { f => f.yCen should be (-0.5) }
    forAll(subfaces.filter(_.normal == -Y)) { f => f.yCen should be (0.5) }
    forAll(subfaces.filter(_.normal == Z)) { f => f.zCen should be (-0.5) }
    forAll(subfaces.filter(_.normal == -Z)) { f => f.zCen should be (0.5) }

  it should "point in correct direction for each location for -x normal" in:
    val face = Face(0, 0, 0, 3, -X)
    val subfaces = face.subdivide()
    forAll(subfaces.filter(_.normal == Y)) { f => f.yCen should be (0.5) }
    forAll(subfaces.filter(_.normal == -Y)) { f => f.yCen should be (-0.5) }
    forAll(subfaces.filter(_.normal == Z)) { f => f.zCen should be (0.5) }
    forAll(subfaces.filter(_.normal == -Z)) { f => f.zCen should be (-0.5) }

  it should "point in correct direction for each location for y normal" in:
    val face = Face(0, 0, 0, 3, Y)
    val subfaces = face.subdivide()
    forAll(subfaces.filter(_.normal == X)) { f => f.xCen should be (-0.5) }
    forAll(subfaces.filter(_.normal == -X)) { f => f.xCen should be (0.5) }
    forAll(subfaces.filter(_.normal == Z)) { f => f.zCen should be (-0.5) }
    forAll(subfaces.filter(_.normal == -Z)) { f => f.zCen should be (0.5) }

  it should "point in correct direction for each location for -y normal" in:
    val face = Face(0, 0, 0, 3, -Y)
    val subfaces = face.subdivide()
    forAll(subfaces.filter(_.normal == X)) { f => f.xCen should be (0.5) }
    forAll(subfaces.filter(_.normal == -X)) { f => f.xCen should be (-0.5) }
    forAll(subfaces.filter(_.normal == Z)) { f => f.zCen should be (0.5) }
    forAll(subfaces.filter(_.normal == -Z)) { f => f.zCen should be (-0.5) }

  it should "point in correct direction for each location for z normal" in:
    val face = Face(0, 0, 0, 3, Z)
    val subfaces = face.subdivide()
    forAll(subfaces.filter(_.normal == Y)) { f => f.yCen should be(-0.5) }
    forAll(subfaces.filter(_.normal == -Y)) { f => f.yCen should be(0.5) }
    forAll(subfaces.filter(_.normal == X)) { f => f.xCen should be(-0.5) }
    forAll(subfaces.filter(_.normal == -X)) { f => f.xCen should be(0.5) }

  it should "point in correct direction for each location for -z normal" in:
    val face = Face(0, 0, 0, 3, -Z)
    val subfaces = face.subdivide()
    forAll(subfaces.filter(_.normal == Y)) { f => f.yCen should be(0.5) }
    forAll(subfaces.filter(_.normal == -Y)) { f => f.yCen should be(-0.5) }
    forAll(subfaces.filter(_.normal == X)) { f => f.xCen should be(0.5) }
    forAll(subfaces.filter(_.normal == -X)) { f => f.xCen should be(-0.5) }

  it should "have correct x positions for x normal" in:
    val face = Face(0, 0, 0, 1, X)
    val subfaces = face.subdivide().filter(_.normal != X)
    withClue(s"xCen: ${subfaces.map(_.xCen)}") {
      forAll(subfaces) { _.xCen should be(-1 / 6f) }
    }

  it should "have correct x positions for -x normal" in:
    val face = Face(0, 0, 0, 1, -X)
    val subfaces = face.subdivide().filter(_.normal != -X)
    withClue(s"xCen: ${subfaces.map(_.xCen)}") {
      forAll(subfaces) { _.xCen should be(1 / 6f) }
    }

  it should "have correct y positions for y normal" in:
    val face = Face(0, 0, 0, 1, Y)
    val subfaces = face.subdivide().filter(_.normal != Y)
    withClue(s"yCen: ${subfaces.map(_.yCen)}") {
      forAll(subfaces) { _.yCen should be(-1 / 6f) }
    }

  it should "have correct y positions for -y normal" in:
    val face = Face(0, 0, 0, 1, -Y)
    val subfaces = face.subdivide().filter(_.normal != -Y)
    withClue(s"yCen: ${subfaces.map(_.yCen)}") {
      forAll(subfaces) { _.yCen should be(1 / 6f) }
    }

  it should "have correct z positions for z normal" in:
    val face = Face(0, 0, 0, 1, Z)
    val subfaces = face.subdivide().filter(_.normal != Z)
    withClue(s"zCen: ${subfaces.map(_.zCen)}") {
      forAll(subfaces) { _.zCen should be(-1 / 6f) }
    }

  it should "have correct z positions for -z normal" in:
    val face = Face(0, 0, 0, 1, -Z)
    val subfaces = face.subdivide().filter(_.normal != -Z)
    withClue(s"zCen: ${subfaces.map(_.zCen)}") {
      forAll(subfaces) { _.zCen should be(1 / 6f) }
    }

  it should "have correct x positions for different scales" in:
    forAll(Seq(X, -X)) { normal =>
      forAll(Seq(1f, 2f, 1 / 3f, 1 / 9f, 0.5f, 1e9f, 1e-9f)) { scale =>
        val face = Face(0, 0, 0, scale, normal)
        val subfaces = face.subdivide().filter(_.normal != normal)
        withClue(s"xCen: ${subfaces.map(_.xCen)}") {
          forAll(subfaces) { _.xCen should be (-normal.sign * scale / 6f) }
        }
      }
    }

  it should "have correct y positions for different scales" in:
    forAll(Seq(Y, -Y)) { normal =>
      forAll(Seq(1f, 2f, 1 / 3f, 1 / 9f, 0.5f, 1e9f, 1e-9f)) { scale =>
        val face = Face(0, 0, 0, scale, normal)
        val subfaces = face.subdivide().filter(_.normal != normal)
        withClue(s"yCen: ${subfaces.map(_.yCen)}") {
          forAll(subfaces) { _.yCen should be (-normal.sign * scale / 6f) }
        }
      }
    }

  it should "have correct z positions for different scales" in:
    forAll(Seq(Z, -Z)) { normal =>
      forAll(Seq(1f, 2f, 1 / 3f, 1 / 9f, 0.5f, 1e9f, 1e-9f)) { scale =>
        val face = Face(0, 0, 0, scale, normal)
        val subfaces = face.subdivide().filter(_.normal != normal)
        withClue(s"zCen: ${subfaces.map(_.zCen)}") {
          forAll(subfaces) { _.zCen should be (-normal.sign * scale / 6f) }
        }
      }
    }

  "subfaces of face with non-zero center" should "keep center equal in normal direction" in:
    val face = Face(0, 0, 1, 1, Z)
    val subfaces = face.subdivide()
    subfaces.filter(_.normal == Z).map(_.zCen) should contain only 1

  "subdividing any face twice" should "give a total of 12*12 subfaces" in:
    forAll(Seq(X, Y, Z, -X, -Y, -Z)) { normal =>
      val face = Face(0, 0, 0, 1, normal)
      val twiceSubdivided = face.subdivide().flatMap(_.subdivide())
      twiceSubdivided should have size 12 * 12
    }

  it should "end up with normals pointing in all directions" in:
    forAll(Seq(X, Y, Z, -X, -Y, -Z)) { normal =>
      val face = Face(0, 0, 0, 1, normal)
      val twiceSubdivided = face.subdivide().flatMap(_.subdivide())
      twiceSubdivided.map(_.normal) should contain only (X, Y, Z, -X, -Y, -Z)
    }

  it should "end up with subfaces 1/9 the original size" in:
    forAll(Seq(X, Y, Z, -X, -Y, -Z)) { normal =>
      val face = Face(0, 0, 0, 1, normal)
      val twiceSubdivided = face.subdivide().flatMap(_.subdivide())
      forAll(twiceSubdivided) {_.scale should be (1f / 9f) }
    }

  "a face's vertices" should "have size 4" in:
    forAll(Seq(X, Y, Z, -X, -Y, -Z)) { normal =>
      val vertices = Face(0, 0, 0, 1, normal).vertices
      vertices.size should be (4)
    }

  it should "be correct for +/-x normal" in:
    forAll(Seq(X, -X)) { normal =>
      val vertices = Face(0, 0, 0, 1, normal).vertices
      vertices.toList.map(_.position) should contain only (
        Vector3(0, -0.5f, -0.5f), Vector3(0, 0.5f, -0.5f),
        Vector3(0, 0.5f, 0.5f), Vector3(0, -0.5f, 0.5f)
      )
    }

  it should "be correct for +/-y normal" in:
    forAll(Seq(Y, -Y)) { normal =>
      val vertices = Face(0, 0, 0, 1, normal).vertices
      vertices.toList.map(_.position) should contain only (
        Vector3(-0.5f, 0, -0.5f), Vector3(0.5f, 0, -0.5f),
        Vector3(0.5f, 0, 0.5f), Vector3(-0.5f, 0, 0.5f)
      )
    }

  it should "be correct for +/-z normal" in:
    forAll(Seq(Z, -Z)) { normal =>
      val vertices = Face(0, 0, 0, 1, normal).vertices
      vertices.toList.map(_.position) should contain only (
        Vector3(-0.5f, -0.5f, 0), Vector3(0.5f, -0.5f, 0),
        Vector3(0.5f, 0.5f, 0), Vector3(-0.5f, 0.5f, 0)
      )
    }

  "vertices of a face" should "be correct for different sizes" in:
    forAll (Seq(1f, 2f, 1e9f, 0.1f, 1e-9f)) { size =>
      val vertices = Face(0, 0, 0, size, X).vertices
      val half = size / 2
      vertices.toList.map(_.position) should contain only(
        Vector3(0, -half, -half), Vector3(0, half, -half),
        Vector3(0, half, half), Vector3(0, -half, half)
      )
    }

  forAll(Seq(
    ( 0f, 0f, 1f), ( 0f, 1f,  0f), ( 1f,  0f, 0f), ( 0f,  0f, -1f), (0f, -1f, 0f), (-1f, 0f, 0f),
    ( 1f, 1f, 1f), ( 1f, 1f, -1f), ( 1f, -1f, 1f), ( 1f, -1f, -1f),
    (-1f, 1f, 1f), (-1f, 1f, -1f), (-1f, -1f, 1f), (-1f, -1f, -1f)
  )) { case (xCen, yCen, zCen) =>
    it should s"be correct for center <$xCen, $yCen, $zCen>" in:
      val half = 0.5f
      val vertices = Face(xCen, yCen, zCen, 1, X).vertices
      vertices.toList.map(_.position) should contain only (
        Vector3(xCen, yCen - half, zCen - half), Vector3(xCen, yCen + half, zCen - half),
        Vector3(xCen, yCen + half, zCen + half), Vector3(xCen, yCen - half, zCen + half)
      )
  }
