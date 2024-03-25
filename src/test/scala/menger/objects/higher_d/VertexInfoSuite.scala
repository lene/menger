package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec

class VertexInfoSuite extends AnyFlatSpec:
  "VertexInfo from Vector3" should "set a position" in:
    val vertex = VertexInfo(Vector3(1, 2, 3))
    assert(vertex.hasPosition)

  it should "keep set position" in:
    val vertex = VertexInfo(Vector3(1, 2, 3))
    assert(vertex.position == Vector3(1, 2, 3))

  it should "not set normal" in:
    val vertex = VertexInfo(Vector3(1, 2, 3))
    assert(!vertex.hasNormal)

  it should "not set color" in:
    val vertex = VertexInfo(Vector3(1, 2, 3))
    assert(!vertex.hasColor)

