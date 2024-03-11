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
