package menger.objects.higher_d

import menger.objects.Vector
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class VectorSuite extends AnyFlatSpec with Matchers:
  "A Vector of any dimension" should "be instantiable" in:
    Vector[1, Float](1)
    Vector[2, Float](1, 2)

  it should "have the correct number of elements" in:
    val v1 = Vector[1, Float](1)
    v1.dimension should be (1)
    v1.v should have size 1

  it should "fail when instantiated with the wrong number of elements" in:
    an [IllegalArgumentException] should be thrownBy Vector[1, Float](1, 2)
    an [IllegalArgumentException] should be thrownBy Vector[2, Float](1)

  it should "instantiate from a Sequence" in:
    val v1 = Vector.fromSeq[2, Float](Seq(1f, 2f))
    v1.dimension should be (2)
    v1.v should contain theSameElementsInOrderAs Seq(1f, 2f)

  it should "fail when instantiated from a Sequence with the wrong number of elements" in:
    an [IllegalArgumentException] should be thrownBy Vector.fromSeq[1, Float](Seq(1f, 2f))
    an [IllegalArgumentException] should be thrownBy Vector.fromSeq[2, Float](Seq(1f))
    an [IllegalArgumentException] should be thrownBy Vector.fromSeq[2, Float](Seq(1f, 2f, 3f))

  "Addition of two Vectors" should "return a Vector with the correct elements" in:
    val v1 = Vector[2, Float](1f, 2f)
    val v2 = Vector[2, Float](3f, 4f)
    (v1 + v2).v should contain theSameElementsInOrderAs Seq(4f, 6f)
  
  "Subtraction of two Vectors" should "return a Vector with the correct elements" in:
    val v1 = Vector[2, Float](3f, 4f)
    val v2 = Vector[2, Float](1f, 2f)
    (v1 - v2).v should contain theSameElementsInOrderAs Seq(2f, 2f)
    
  "Test for equality" should "work correctly" in:
    val v1 = Vector[2, Float](1f, 2f)
    val v2 = Vector[2, Float](1f, 2f)
    val v3 = Vector[2, Float](2f, 3f)
    
    (v1 === v2) should be (true)
    (v1 === v3) should be (false)
    
  "Length calculation" should "return the correct length" in:
    val v1 = Vector[2, Float](3f, 4f)
    v1.len() should be (5f) // 3^2 + 4^2 = 25, sqrt(25) = 5
    
  "unary -" should "work correctly" in:
    val v1 = Vector[2, Float](1f, 2f)
    (-v1).v should contain theSameElementsInOrderAs Seq(-1f, -2f)
    
  "dot product" should "return the correct value" in:
    val v1 = Vector[2, Float](1f, 2f)
    val v2 = Vector[2, Float](3f, 4f)
    v1.dot(v2) should be (11f) // 1*3 + 2*4 = 3 + 8 = 11
    
  "dst" should "return the correct distance" in:
    val v1 = Vector[2, Float](1f, 2f)
    val v2 = Vector[2, Float](4f, 6f)
    v1.dst(v2) should be (5f) // sqrt((4-1)^2 + (6-2)^2) = sqrt(9 + 16) = sqrt(25) = 5
    
  "dst2" should "return the correct squared distance" in:
    val v1 = Vector[2, Float](1f, 2f)
    val v2 = Vector[2, Float](4f, 6f)
    v1.dst2(v2) should be (25f) // (4-1)^2 + (6-2)^2 = 9 + 16 = 25
    
  "Zero Vector" should "be correctly instantiated" in:
    val zeroVec = Vector.Zero[3, Float]
    zeroVec.v should contain theSameElementsInOrderAs Seq(0f, 0f, 0f)
    zeroVec.dimension should be (3)
    
  "toArray" should "return the correct array representation" in:
    val v1 = Vector[2, Float](1f, 2f)
    v1.toArray should contain theSameElementsInOrderAs Array(1f, 2f)
    