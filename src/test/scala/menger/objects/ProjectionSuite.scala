package menger.objects

import com.badlogic.gdx.math.{Vector3, Vector4}
import menger.objects.higher_d.{Projection, VertexInfo}
import org.scalatest.funsuite.AnyFunSuite

class ProjectionSuite extends AnyFunSuite:
  test("instantiate a valid projection") {
    Projection(2, 1)
  }

  test("instantiating a projection with screen behind camera fails") {
    assertThrows[AssertionError] {
      Projection(1, 2)
    }
  }

  test("instantiating a projection with screen on same distance as camera fails") {
    assertThrows[AssertionError] {
      Projection(1, 1)
    }
  }

  test("instantiating a projection with negative screen W fails") {
    assertThrows[AssertionError] {
      Projection(2, -1)
    }
  }

  test("instantiating a projection with negative camera W fails") {
    assertThrows[AssertionError] {
      Projection(-1, -2)
    }
  }

  test("projecting a point") {
    val projection = Projection(2, 1)
    val point = Vector4(1, 0, 0, 0)
    val projected = projection(point)
    assert(projected.x == 0.5f && projected.y == 0.0f && projected.z == 0.0f)
  }

  test("projecting a point list") {
    val projection = Projection(2, 1)
    val points = Seq(Vector4(1, 0, 0, 0), Vector4(1, 0, 0, 0))
    val projected = projection(points)
    projected.foreach(p => assert(p.x == 0.5f && p.y == 0.0f && p.z == 0.0f))
  }

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
    