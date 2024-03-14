package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import org.scalatest.funsuite.AnyFunSuite

class RotationSuite extends AnyFunSuite:
  test("zero Rotation keeps point same") {
    val p = Vector4(1, 2, 3, 4)
    val r = Rotation(0, 0, 0)
    assert(r(p) == p)
  }
  
  test("zero Rotation keeps Tesseract same") {
    val t = Tesseract(1)
    val r = Rotation(0, 0, 0)
    assert(r(t.vertices) == t.vertices)
  }

  test("value outside [0, 360) is mapped to [0, 360)") {
    assert(Rotation(360, 0, 0).rotXW == 0)
  }

  test("rotations with multiples of 360° are zero") {
    Seq(0f, 360f, 720f, -1080f).foreach { angle =>
      assert(Rotation(angle, 0, 0).isZero)
      assert(Rotation(0, angle, 0).isZero)
      assert(Rotation(angle, angle, angle).isZero)
    }
  }

  test("positive value > 360°") {
    assert(Rotation(370, 0, 0).rotXW == 10)
  }

  test("negative value") {
    assert(Rotation(-10, 0, 0).rotXW == 350)
  }

  test("adding two rotations with sum < 360°") {
    val r1 = Rotation(10, 0, 0)
    val r2 = Rotation(20, 0, 0)
    assert((r1 + r2).rotXW == 30)
  }

  test("adding two rotations with sum > 360°") {
    val r1 = Rotation(350, 0, 0)
    val r2 = Rotation(20, 0, 0)
    assert((r1 + r2).rotXW == 10)
  }

  test("adding two rotations with sum = 360°") {
    val r1 = Rotation(350, 0, 0)
    val r2 = Rotation(10, 0, 0)
    assert((r1 + r2).isZero)
  }

  test("adding two rotations has same result as applying them in sequence") {
    val r1 = Rotation(10, 0, 0)
    val r2 = Rotation(20, 0, 0)
    val p = Vector4(1, 2, 3, 4)
    assertEqual((r1 + r2)(p), r2(r1(p)))
  }

  test("rotating 90 degrees around xw plane") {
    val r = Rotation(90, 0, 0)
    assertEqual(r(Vector4(1, 0, 0, 0)), Vector4(0, 0, 0, -1))
  }

  test("rotating 90 degrees around yw plane") {
    val r = Rotation(0, 90, 0)
    assertEqual(r(Vector4(0, 1, 0, 0)), Vector4(0, 0, 0, -1))
  }

  test("rotating 90 degrees around zw plane") {
    val r = Rotation(0, 0, 90)
    assertEqual(r(Vector4(0, 0, 1, 0)), Vector4(0, 0, 0, -1))
  }


def assertEqual(v1: Vector4, v2: Vector4, epsilon: Float = 1e-16): Unit =
  assert(v1.dst(v2) < epsilon, s"$v1 != $v2")
