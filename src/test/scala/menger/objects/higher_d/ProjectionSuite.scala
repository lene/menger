package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
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

  test("adding a projection with bigger eyeW increases eyeW") {
    val p1 = Projection(2, 1)
    val p2 = Projection(3, 1)
    val p3 = p1 + p2
    assert(p3.eyeW > p1.eyeW)
  }

  test("adding a projection with bigger eyeW sets eyeW to less than sum of eyeWs") {
    val p1 = Projection(2, 1)
    val p2 = Projection(3, 1)
    val p3 = p1 + p2
    assert(p3.eyeW < p1.eyeW + p2.eyeW)
  }

  test("adding a projection with bigger eyeW increases eyeW by exponent defined in class") {
    val p1 = Projection(2, 1)
    val p2 = Projection(3, 1)
    val p3 = p1 + p2
    assertResult(p3.eyeW === math.pow(p1.eyeW, p1.addExponent))
  }

  test("adding a projection with smaller eyeW decreases eyeW") {
    val p1 = Projection(3, 1)
    val p2 = Projection(2, 1)
    val p3 = p1 + p2
    assert(p3.eyeW < p1.eyeW)
  }

  test("adding a projection with smaller eyeW sets eyeW to less than sum of eyeWs") {
    val p1 = Projection(3, 1)
    val p2 = Projection(2, 1)
    val p3 = p1 + p2
    assert(p3.eyeW < p1.eyeW + p2.eyeW)
  }

  test("adding a projection with smaller eyeW decreases eyeW by exponent defined in class") {
    val p1 = Projection(3, 1)
    val p2 = Projection(2, 1)
    val p3 = p1 + p2
    assertResult(p3.eyeW === math.pow(p1.eyeW, 1f / p1.addExponent))
  }

  test("adding a projection with same eyeW keeps eyeW equal") {
    val p1 = Projection(3, 1)
    val p2 = Projection(3, 1)
    val p3 = p1 + p2
    assert(p3.eyeW == p1.eyeW)
  }

  test("adding a projection keeps screenW equal") {
    val p1 = Projection(3, 1)
    val p2 = Projection(2, 1)
    val p3 = p1 + p2
    assert(p3.screenW == p1.screenW)
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
