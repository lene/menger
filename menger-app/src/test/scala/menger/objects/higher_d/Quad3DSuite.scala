package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class Quad3DSuite extends AnyFlatSpec with Matchers:
  "A RectVertices3D" should "be able to be created with 4 vertices" in:
    Quad3D(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 1, 0), Vector3(0, 1, 0))

  it should "not be constrained to be square" in:
    Quad3D(Vector3(0, 0, 0), Vector3(2, 0, 0), Vector3(1, 1, 0), Vector3(0, 1, 0))

  it should "have a dimension of 4" in:
    Quad3D(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 1, 0), Vector3(0, 1, 0)).dimension should be(4)