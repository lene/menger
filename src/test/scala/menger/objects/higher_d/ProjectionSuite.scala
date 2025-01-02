package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import org.scalatest.Assertions.assertThrows
import org.scalatest.flatspec.AnyFlatSpec

class ProjectionSuite extends AnyFlatSpec:
  "instantiating a projection" should "be valid if eyeW > screenW" in:
    Projection(2, 1)

  it should "fail if screen behind camera" in:
    assertThrows[IllegalArgumentException] {Projection(1, 2)}

  it should "fail if screen on same distance as camera" in:
    assertThrows[IllegalArgumentException] {Projection(1, 1)}

  it should "fail if screen W negative" in:
    assertThrows[IllegalArgumentException] {Projection(2, -1)}

  it should "fail if camera W negative" in:
    assertThrows[IllegalArgumentException] {Projection(-1, -2)}

  "adding a projection with bigger eyeW" should "increase eyeW" in:
    val p1 = Projection(2, 1)
    val p2 = Projection(3, 1)
    val p3 = p1 + p2
    assert(p3.eyeW > p1.eyeW)

  it should "set eyeW to less than sum of eyeWs" in:
    val p1 = Projection(2, 1)
    val p2 = Projection(3, 1)
    val p3 = p1 + p2
    assert(p3.eyeW < p1.eyeW + p2.eyeW)

  it should "increase eyeW by exponent defined in class" in:
    val p1 = Projection(2, 1)
    val p2 = Projection(3, 1)
    val p3 = p1 + p2
    assertResult(p3.eyeW === math.pow(p1.eyeW, p1.addExponent))

  "adding a projection with smaller eyeW" should "decrease eyeW" in:
    val p1 = Projection(3, 1)
    val p2 = Projection(2, 1)
    val p3 = p1 + p2
    assert(p3.eyeW < p1.eyeW)

  it should "set eyeW to less than sum of eyeWs" in:
    val p1 = Projection(3, 1)
    val p2 = Projection(2, 1)
    val p3 = p1 + p2
    assert(p3.eyeW < p1.eyeW + p2.eyeW)

  it should "decrease eyeW by exponent defined in class" in:
    val p1 = Projection(3, 1)
    val p2 = Projection(2, 1)
    val p3 = p1 + p2
    assertResult(p3.eyeW === math.pow(p1.eyeW, 1f / p1.addExponent))

  "adding a projection with same eyeW" should "keep eyeW equal" in:
    val p1 = Projection(3, 1)
    val p2 = Projection(3, 1)
    val p3 = p1 + p2
    assert(p3.eyeW == p1.eyeW)

  "adding a projection" should "keep screenW equal" in:
    val p1 = Projection(3, 1)
    val p2 = Projection(2, 1)
    val p3 = p1 + p2
    assert(p3.screenW == p1.screenW)

  "projecting a point" should "give expected result" in:
    val projection = Projection(2, 1)
    val point = Vector4(1, 0, 0, 0)
    val projected = projection(point)
    assert(projected.x == 0.5f && projected.y == 0.0f && projected.z == 0.0f)

  "projecting a point list" should "give expected result" in:
    val projection = Projection(2, 1)
    val points = Seq(Vector4(1, 0, 0, 0), Vector4(1, 0, 0, 0))
    val projected = projection(points)
    projected.foreach(p => assert(p.x == 0.5f && p.y == 0.0f && p.z == 0.0f))
