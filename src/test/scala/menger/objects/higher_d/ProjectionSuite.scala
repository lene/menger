package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Inspectors.forAll

class ProjectionSuite extends AnyFlatSpec with Matchers:
  "instantiating a projection" should "be valid if eyeW > screenW" in:
    Projection(2, 1)

  it should "fail if screen behind camera" in:
    an [IllegalArgumentException] should be thrownBy Projection(1, 2)

  it should "fail if screen on same distance as camera" in:
    an [IllegalArgumentException] should be thrownBy Projection(1, 1)

  it should "fail if screen W negative" in:
    an [IllegalArgumentException] should be thrownBy Projection(2, -1)

  it should "fail if camera W negative" in:
    an [IllegalArgumentException] should be thrownBy Projection(-1, -2)

  "adding a projection with bigger eyeW" should "increase eyeW" in:
    val p1 = Projection(2, 1)
    val p2 = Projection(3, 1)
    val p3 = p1 + p2
    p3.eyeW should be > p1.eyeW

  it should "set eyeW to less than sum of eyeWs" in:
    val p1 = Projection(2, 1)
    val p2 = Projection(3, 1)
    val p3 = p1 + p2
    p3.eyeW should be < p1.eyeW + p2.eyeW

  it should "increase eyeW by exponent defined in class" in:
    val p1 = Projection(2, 1)
    val p2 = Projection(3, 1)
    val p3 = p1 + p2
    p3.eyeW should equal (math.pow(p1.eyeW, p1.addExponent).toFloat +- Const.epsilon)

  "adding a projection with smaller eyeW" should "decrease eyeW" in:
    val p1 = Projection(3, 1)
    val p2 = Projection(2, 1)
    val p3 = p1 + p2
    p3.eyeW should be < p1.eyeW

  it should "set eyeW to less than sum of eyeWs" in:
    val p1 = Projection(3, 1)
    val p2 = Projection(2, 1)
    val p3 = p1 + p2
    p3.eyeW should be < p1.eyeW + p2.eyeW

  it should "decrease eyeW by exponent defined in class" in:
    val p1 = Projection(3, 1)
    val p2 = Projection(2, 1)
    val p3 = p1 + p2
    p3.eyeW should equal (math.pow(p1.eyeW, 1f / p1.addExponent).toFloat +- Const.epsilon)

  "adding a projection with same eyeW" should "keep eyeW equal" in:
    val p1 = Projection(3, 1)
    val p2 = Projection(3, 1)
    val p3 = p1 + p2
    p3.eyeW should equal (p1.eyeW)

  "adding a projection" should "keep screenW equal" in:
    val p1 = Projection(3, 1)
    val p2 = Projection(2, 1)
    val p3 = p1 + p2
    p3.screenW should equal (p1.screenW)

  "projecting a point" should "give expected result" in:
    val projection = Projection(2, 1)
    val point = Vector4(1, 0, 0, 0)
    val projected = projection(point)
    projected.x should equal (0.5f)
    projected.y should equal (0.0f)
    projected.z should equal (0.0f)

  "projecting a point list" should "give expected result" in:
    val projection = Projection(2, 1)
    val points = Seq(Vector4(1, 0, 0, 0), Vector4(1, 0, 0, 0))
    val projected = projection(points)
    forAll(projected) { p =>
      p.x should equal(0.5f)
      p.y should equal(0.0f)
      p.z should equal(0.0f)
    }
