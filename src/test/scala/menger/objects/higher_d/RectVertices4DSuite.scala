package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import com.badlogic.gdx.math.Vector4

class RectVertices4DSuite extends AnyFlatSpec with RectMesh with Matchers:

  "Normals with collinear vectors" should "fail" in:
    Seq(Vector4.X, Vector4.Y, Vector4.Z, Vector4.W).foreach(vec =>
      assertThrows[IllegalArgumentException](normals(Seq(vec, vec)))
    )

  Seq(Vector4.X, Vector4.Y, Vector4.Z, Vector4.W).combinations(2).foreach { case Seq(vec1, vec2) =>
    s"Normals to $vec1 and $vec2" should "be orthogonal to both" in:
      val n = normals(Seq(vec1, vec2))
      assert(n.forall(_.dot(vec1) == 0))
      assert(n.forall(_.dot(vec2) == 0))
  }

  "Faces created with collinear vectors" should "fail" in:
    Seq(Vector4.X, Vector4.Y, Vector4.Z, Vector4.W).foreach(vec =>
      assertThrows[IllegalArgumentException](RectVertices4D(Vector4.Zero, 1, Seq(vec, vec)))
    )

  "A Face in xy direction" should "be instantiated from center, scale and normal" in:
    assert(RectVertices4D(Vector4.Zero, 1, Seq(Vector4.Z, Vector4.W)) == RectVertices4D(
      Vector4(-1, -1, 0, 0), Vector4(1, -1, 0, 0), Vector4(1, 1, 0, 0), Vector4(-1, 1, 0, 0)
    ))
