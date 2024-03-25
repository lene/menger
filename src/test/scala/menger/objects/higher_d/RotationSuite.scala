package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import org.scalatest.flatspec.AnyFlatSpec

class RotationSuite extends AnyFlatSpec:
  "zero Rotation" should "keep point same" in:
    val p = Vector4(1, 2, 3, 4)
    val r = Rotation(0, 0, 0)
    assert(r(p) == p)
  
  it should "keep Tesseract same" in:
    val t = Tesseract(1)
    val r = Rotation(0, 0, 0)
    assert(r(t.vertices) == t.vertices)

  "value outside [0, 360)" should "be mapped to 0 if equals 360" in:
    assert(Rotation(360, 0, 0).rotXW == 0)

  it should "be mapped to 0 if multiples of 360°" in:
    Seq(0f, 360f, 720f, -1080f).foreach { angle =>
      assert(Rotation(angle, 0, 0).isZero)
      assert(Rotation(0, angle, 0).isZero)
      assert(Rotation(angle, angle, angle).isZero)
    }

  it should "be positive if > 360°" in:
    assert(Rotation(370, 0, 0).rotXW == 10)

  it should "be positive if < 0°" in:
    assert(Rotation(-10, 0, 0).rotXW == 350)

  "adding two rotations" should "be positive if sum < 360°" in:
    val r1 = Rotation(10, 0, 0)
    val r2 = Rotation(20, 0, 0)
    assert((r1 + r2).rotXW == 30)

  it should "be positive if sum > 360°" in:
    val r1 = Rotation(350, 0, 0)
    val r2 = Rotation(20, 0, 0)
    assert((r1 + r2).rotXW == 10)

  it should "be zero if sum == 360°" in:
    val r1 = Rotation(350, 0, 0)
    val r2 = Rotation(10, 0, 0)
    assert((r1 + r2).isZero)

  it should "have same result as applying them in sequence" in:
    val r1 = Rotation(10, 0, 0)
    val r2 = Rotation(20, 0, 0)
    val p = Vector4(1, 2, 3, 4)
    assertEqual((r1 + r2)(p), r2(r1(p)))

  "rotating 90 degrees" should "work around xw plane" in:
    val r = Rotation(90, 0, 0)
    assertEqual(r(Vector4(1, 0, 0, 0)), Vector4(0, 0, 0, -1))

  it should "work around yw plane" in:
    val r = Rotation(0, 90, 0)
    assertEqual(r(Vector4(0, 1, 0, 0)), Vector4(0, 0, 0, -1))

  it should "work around zw plane" in:
    val r = Rotation(0, 0, 90)
    assertEqual(r(Vector4(0, 0, 1, 0)), Vector4(0, 0, 0, -1))

def assertEqual(v1: Vector4, v2: Vector4, epsilon: Float = 1e-16): Unit =
  assert(v1.dst(v2) < epsilon, s"$v1 != $v2")
