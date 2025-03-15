package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import com.badlogic.gdx.math.{Matrix4, Vector4}

class MatrixSuite extends AnyFlatSpec with Matchers:
  val rawValues: Array[Float] = Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
  "Matrix string representation" should "have 4 lines" in:
    val m = Matrix4(rawValues)
    m.toString.split("\n") should have size 4

  it should "have every element represented" in:
    val m = Matrix4(rawValues)
    rawValues.map(_.toString).foreach(v => m.toString should include(v))
