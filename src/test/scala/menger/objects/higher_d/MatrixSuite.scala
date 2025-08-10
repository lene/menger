package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.objects.{Matrix, Vector}
import CustomMatchers.*

class MatrixSuite extends AnyFlatSpec with Matchers:
  val rawValues: Array[Float] = Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
  "A Matrix" should "be creatable from an array of values" in:
    val m = Matrix[4](rawValues)
    m.m should contain theSameElementsInOrderAs rawValues

  it should "have the correct dimension 4" in:
    val m = Matrix[4](rawValues)
    m.dimension should be(4)

  it should "have the correct dimension 3" in:
    val m = Matrix[3](rawValues.take(9))
    m.dimension should be(3)

  it should "work with dimension 1" in:
    val m = Matrix[1](Array(1f))
    m.m should contain theSameElementsInOrderAs Array(1f)
    m.dimension should be(1)

  it should "not work with negative dimension" in:
    an [IllegalArgumentException] should be thrownBy Matrix[-1](rawValues)

  it should "not be creatable with an incorrect number of values" in:
    an [IllegalArgumentException] should be thrownBy Matrix[4](Array(1, 2, 3))
    an [IllegalArgumentException] should be thrownBy Matrix[3](rawValues)
    an [IllegalArgumentException] should be thrownBy Matrix[1](Array())

  "Matrix string representation" should "have correct number lines for dimension 4" in:
    val m = Matrix[4](rawValues)
    m.toString.split("\n") should have size 4

  it should "have correct number lines for dimension 2" in:
    val m = Matrix[2](rawValues.take(4))
    m.toString.split("\n") should have size 2

  it should "have every element represented" in:
    val m = Matrix[4](rawValues)
    rawValues.map(_.toString).foreach(v => m.toString should include(v))

  "Identity" should "be a matrix of correct size" in:
    val identity = Matrix.identity[4]
    identity.m should have size 16

  it should "have 1s on the diagonal" in:
    val identity = Matrix.identity[4]
    (0 until 4).foreach { i => identity(i, i) should be(1f) }

  it should "have 0s elsewhere" in:
    val identity = Matrix.identity[4]
    for (i <- 0 until 4; j <- 0 until 4 if i != j) {
      identity(i, j) should be(0f)
    }

  it should "leave a vector unchanged when multiplied" in:
    val identity = Matrix.identity[4]
    val v = Vector[4](1f, 2f, 3f, 4f)
    val result = identity(v)
    result should epsilonEqual(v)

  "Matrix multiplication" should "multiply identity with itself" in:
    val identity = Matrix.identity[4]
    val result = identity * identity
    result should epsilonEqual(identity)

  it should "multiply identity with another matrix" in:
    val m = Matrix[4](rawValues)
    val identity = Matrix.identity[4]
    val result = identity * m
    result should epsilonEqual(m)

  it should "multiply another matrix with identity" in:
    val m = Matrix[4](rawValues)
    val identity = Matrix.identity[4]
    val result = m * identity
    result should epsilonEqual(m)

  it should "lead to a different Matrix if identity is not involved" in:
    val m1 = Matrix[4](rawValues)
    val m2 = Matrix[4](rawValues.reverse)
    val result = m1 * m2
    result should not be epsilonEqual(m1)
    result should not be epsilonEqual(m2)

  it should "not be commutative"  in:
    val m1 = Matrix[4](rawValues)
    val m2 = Matrix[4](rawValues.reverse)
    val result1 = m1 * m2
    val result2 = m2 * m1
    result1 should not be epsilonEqual(result2)

  it should "be zero if multiplying with zero matrix" in:
    val m = Matrix[2](rawValues.take(4))
    val zero = Matrix[2](Array.fill(4)(0f))
    val result = m * zero
    result.m should contain only 0f

  "index calculation" should "return the correct index for a 4x4 matrix" in:
    val m = Matrix[4](rawValues)
    m.index(0, 0) should be(0)
    m.index(1, 2) should be(6)
    m.index(3, 3) should be(15)
    
  it should "fail if outside bounds" in:
    val m = Matrix[4](rawValues)
    an [IllegalArgumentException] should be thrownBy m.index(-1, 0)
    an [IllegalArgumentException] should be thrownBy m.index(0, -1)
    an [IllegalArgumentException] should be thrownBy m.index(4, 0)
    an [IllegalArgumentException] should be thrownBy m.index(0, 4)
    
  "accessing an element" should "return the correct value" in:
    val m = Matrix[4](rawValues)
    m(0, 0) should be(1f)
    m(1, 2) should be(7f)
    m(3, 3) should be(16f)
    
  it should "fail if outside bounds"  in:  
    val m = Matrix[4](rawValues)
    an [IllegalArgumentException] should be thrownBy m(-1, 0)
    an [IllegalArgumentException] should be thrownBy m(0, -1)
    an [IllegalArgumentException] should be thrownBy m(4, 0)
    an [IllegalArgumentException] should be thrownBy m(0, 4)
