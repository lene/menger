package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class VertexInfoSuite extends AnyFlatSpec with Matchers:
  "VertexInfo from Vector3" should "set a position" in:
    val vertex = VertexInfo(Vector3(1, 2, 3))
    vertex.hasPosition should be (true)

  it should "keep set position" in:
    val vertex = VertexInfo(Vector3(1, 2, 3))
    vertex.position should be (Vector3(1, 2, 3))

  it should "not set normal" in:
    val vertex = VertexInfo(Vector3(1, 2, 3))
    vertex.hasNormal should be (false)

  it should "not set color" in:
    val vertex = VertexInfo(Vector3(1, 2, 3))
    vertex.hasColor should be (false)

