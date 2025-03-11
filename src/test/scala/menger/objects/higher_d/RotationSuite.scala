package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should._
import CustomMatchers._

class RotationSuite extends AnyFlatSpec with Matchers:
  "zero Rotation" should "keep point same" in:
    val p = Vector4(1, 2, 3, 4)
    val r = Rotation(0, 0, 0)
    r(p) should be (p)
  
  it should "keep Tesseract same" in:
    val t = Tesseract(1)
    val r = Rotation(0, 0, 0)
    r(t.vertices) should be (t.vertices)

  "value outside [0, 360)" should "be mapped to 0 if equals 360" in:
    Rotation(360, 0, 0).rotXW should be (0)

  it should "be mapped to 0 if multiples of 360°" in:
    Seq(0f, 360f, 720f, -1080f).foreach { angle =>
      Rotation(angle, 0, 0).isZero should be (true)
      Rotation(0, angle, 0).isZero should be (true)
      Rotation(angle, angle, angle).isZero should be (true)
    }

  it should "be positive if > 360°" in:
    Rotation(370, 0, 0).rotXW should be (10)

  it should "be positive if < 0°" in:
    Rotation(-10, 0, 0).rotXW should be(350)

  "adding two rotations" should "be positive if sum < 360°" in:
    val r1 = Rotation(10, 0, 0)
    val r2 = Rotation(20, 0, 0)
    (r1 + r2).rotXW should be (30)

  it should "be positive if sum > 360°" in:
    val r1 = Rotation(350, 0, 0)
    val r2 = Rotation(20, 0, 0)
    (r1 + r2).rotXW should be (10)

  it should "be zero if sum == 360°" in:
    val r1 = Rotation(350, 0, 0)
    val r2 = Rotation(10, 0, 0)
    (r1 + r2).isZero should be (true)

  it should "have same result as applying them in sequence" in:
    val r1 = Rotation(10, 0, 0)
    val r2 = Rotation(20, 0, 0)
    val p = Vector4(1, 2, 3, 4)
    (r1 + r2)(p) should be (r2(r1(p)))

  "rotating 90 degrees" should "work around xw plane" in:
    val r = Rotation(90, 0, 0)
    r(Vector4(1, 0, 0, 0)) should epsilonEqual (Vector4(0, 0, 0, -1))

  it should "work around yw plane" in:
    val r = Rotation(0, 90, 0)
    r(Vector4(0, 1, 0, 0)) should epsilonEqual (Vector4(0, 0, 0, -1))

  it should "work around zw plane" in:
    val r = Rotation(0, 0, 90)
    r(Vector4(0, 0, 1, 0)) should epsilonEqual (Vector4(0, 0, 0, -1))
