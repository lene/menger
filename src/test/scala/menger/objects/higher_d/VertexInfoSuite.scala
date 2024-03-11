package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import org.scalatest.funsuite.AnyFunSuite

class VertexInfoSuite extends AnyFunSuite:
  test("VertexInfo from Vector3 sets position") {
    val vertex = VertexInfo(Vector3(1, 2, 3))
    assert(vertex.hasPosition)
  }

  test("VertexInfo from Vector3 keeps set position") {
    val vertex = VertexInfo(Vector3(1, 2, 3))
    assert(vertex.position == Vector3(1, 2, 3))
  }

  test("VertexInfo from Vector3 does not set normal") {
    val vertex = VertexInfo(Vector3(1, 2, 3))
    assert(!vertex.hasNormal)
  }

  test("VertexInfo from Vector3 does not set color") {
    val vertex = VertexInfo(Vector3(1, 2, 3))
    assert(!vertex.hasColor)
  }

