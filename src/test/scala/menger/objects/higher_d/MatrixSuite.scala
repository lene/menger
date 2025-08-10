package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.objects.{Matrix, Vector}
import CustomMatchers.*

class MatrixSuite extends AnyFlatSpec with Matchers:
  val rawValues: Array[Float] = Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
  "Matrix string representation" should "have 4 lines" in:
    val m = Matrix[4](rawValues)
    m.toString.split("\n") should have size 4

  it should "have every element represented" in:
    val m = Matrix[4](rawValues)
    rawValues.map(_.toString).foreach(v => m.toString should include(v))

  "Identity" should "be a 4x4 matrix" in:
    val identity = Matrix.identity[4]
    identity.m should have size 16

  it should "have 1s on the diagonal" in:
    val identity = Matrix.identity[4]
    for (i <- 0 until 4) {
      identity.m(i * 5) should be(1f)
    }

  it should "have 0s elsewhere" in:
    val identity = Matrix.identity[4]
    for (i <- 0 until 4; j <- 0 until 4 if i != j) {
      identity.m(i * 4 + j) should be(0f)
    }

  it should "leave a vector unchanged when multiplied" in:
    val identity = Matrix.identity[4]
    val v = Vector[4](1f, 2f, 3f, 4f)
    val result = identity(v)
    result should epsilonEqual(v)

  "Matrix multiplication" should "multiply identity with itself" in:
    val identity = Matrix.identity[4]
    val result = identity.mul(identity)
    result should epsilonEqual(identity)

  it should "multiply identity with another matrix" in:
    val m = Matrix[4](rawValues)
    val identity = Matrix.identity[4]
    val result = identity.mul(m)
    result should epsilonEqual(m)

  it should "multiply another matrix with identity" in:
    val m = Matrix[4](rawValues)
    val identity = Matrix.identity[4]
    val result = m.mul(identity)
    result should epsilonEqual(m)
