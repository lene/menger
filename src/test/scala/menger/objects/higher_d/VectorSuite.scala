package menger.objects.higher_d

import menger.common.Vector
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class VectorSuite extends AnyFlatSpec with Matchers:
  "A Vector of any dimension" should "be instantiable" in:
    Vector[1](1)
    Vector[2](1, 2)

  it should "have the correct number of elements" in:
    val v1 = Vector[1](1)
    v1.dimension should be (1)
    v1.v should have size 1

  it should "fail when instantiated with the wrong number of elements" in:
    an [IllegalArgumentException] should be thrownBy Vector[1](1, 2)
    an [IllegalArgumentException] should be thrownBy Vector[2](1)

  it should "instantiate from a Sequence" in:
    val v1 = Vector.fromSeq[2](Seq(1f, 2f))
    v1.dimension should be (2)
    v1.v should contain theSameElementsInOrderAs Seq(1f, 2f)

  it should "fail when instantiated from a Sequence with the wrong number of elements" in:
    an [IllegalArgumentException] should be thrownBy Vector.fromSeq[1](Seq(1f, 2f))
    an [IllegalArgumentException] should be thrownBy Vector.fromSeq[2](Seq(1f))
    an [IllegalArgumentException] should be thrownBy Vector.fromSeq[2](Seq(1f, 2f, 3f))

  "Element access" should "work for valid indices" in:
    val v = Vector[3](1f, 2f, 3f)
    v(0) should be (1f)
    v(1) should be (2f)
    v(2) should be (3f)

  it should "throw IndexOutOfBoundsException for invalid indices" in:
    val v = Vector[2](1f, 2f)
    an [IllegalArgumentException] should be thrownBy v(-1)
    an [IllegalArgumentException] should be thrownBy v(2)
    an [IllegalArgumentException] should be thrownBy v(10)

  "Addition of two Vectors" should "return a Vector with the correct elements" in:
    val v1 = Vector[2](1f, 2f)
    val v2 = Vector[2](3f, 4f)
    (v1 + v2).v should contain theSameElementsInOrderAs Seq(4f, 6f)

  "Subtraction of two Vectors" should "return a Vector with the correct elements" in:
    val v1 = Vector[2](3f, 4f)
    val v2 = Vector[2](1f, 2f)
    (v1 - v2).v should contain theSameElementsInOrderAs Seq(2f, 2f)

  "unary -" should "work correctly" in :
    val v1 = Vector[2](1f, 2f)
    (-v1).v should contain theSameElementsInOrderAs Seq(-1f, -2f)

  "Test for approximate equality" should "work correctly" in:
    val v1 = Vector[2](1f, 2f)
    val v2 = Vector[2](1f, 2f)
    val v3 = Vector[2](2f, 3f)
    (v1 === v2) should be (true)
    (v1 === v3) should be (false)

  it should "work with very small values" in:
    val v1 = Vector[2](0.0001f, 0.0002f)
    val v2 = Vector[2](0.0001f + 1e-6f, 0.0002f + 1e-6f)
    v1.epsilonEquals(v2, 1e-5f) should be(true)
    v1.epsilonEquals(v2, 1e-7f) should be(false)

  "Length calculation" should "return the correct length" in:
    val v1 = Vector[2](3f, 4f)
    v1.len should be (5f) // 3^2 + 4^2 = 25, sqrt(25) = 5

  "Scalar multiplication" should "work correctly" in:
    val v1 = Vector[3](1f, 2f, 3f)
    val result = v1 * 2f
    result.v should contain theSameElementsInOrderAs Seq(2f, 4f, 6f)

  "Scalar division" should "work correctly" in:
    val v1 = Vector[3](2f, 4f, 6f)
    val result = v1 / 2f
    result.v should contain theSameElementsInOrderAs Seq(1f, 2f, 3f)

  it should "return infinity when dividing by zero" in:
    val v1 = Vector[2](1f, 2f)
    val result = v1 / 0f
    result.v.forall(_.isInfinite) should be (true)

  "dst" should "return the correct distance" in:
    val v1 = Vector[2](1f, 2f)
    val v2 = Vector[2](4f, 6f)
    v1.dst(v2) should be (5f) // sqrt((4-1)^2 + (6-2)^2) = sqrt(9 + 16) = sqrt(25) = 5

  "dst2" should "return the correct squared distance" in:
    val v1 = Vector[2](1f, 2f)
    val v2 = Vector[2](4f, 6f)
    v1.dst2(v2) should be (25f) // (4-1)^2 + (6-2)^2 = 9 + 16 = 25

  "dot product" should "return the correct value" in :
    val v1 = Vector[2](1f, 2f)
    val v2 = Vector[2](3f, 4f)
    v1 * v2 should be(11f) // 1*3 + 2*4 = 3 + 8 = 11

  "Zero Vector" should "be correctly instantiated" in:
    val zeroVec = Vector.Zero[3]
    zeroVec.v should contain theSameElementsInOrderAs Seq(0f, 0f, 0f)
    zeroVec.dimension should be (3)

  "toString" should "format correctly" in:
    val v1 = Vector[3](1f, 2.5f, 3f)
    v1.toString should include ("1")
    v1.toString should include ("2.50")
    v1.toString should include ("3")
    v1.toString should startWith ("<")
    v1.toString should endWith (">")

  "count" should "work correctly" in:
    val v = Vector[4](1f, 2f, 3f, 4f)
    v.count(_ > 2f) should be (2)

  "filter" should "work correctly" in :
    val v = Vector[4](1f, 2f, 3f, 4f)
    v.filter(_ > 2f) should contain theSameElementsAs Seq(3f, 4f)

  "forall" should "work correctly" in :
    val v = Vector[4](1f, 2f, 3f, 4f)
    v.forall(_ > 0f) should be (true)
    v.forall(_ > 2f) should be (false)

  "indexWhere" should "work correctly" in :
    val v = Vector[4](1f, 2f, 3f, 4f)
    v.indexWhere(_ > 2f) should be (2)

  "map" should "work correctly" in :
    val v = Vector[4](1f, 2f, 3f, 4f)
    v.map(_ * 2) should contain theSameElementsInOrderAs Seq(2f, 4f, 6f, 8f)

  "unit vector" should "be correctly instantiated" in:
    val unitVec = Vector.unit[3](1)
    unitVec.v should contain theSameElementsInOrderAs Seq(0f, 1f, 0f)
    unitVec.dimension should be (3)

  it should "fail for invalid direction" in:
    an [IllegalArgumentException] should be thrownBy Vector.unit[2](-1)
    an [IllegalArgumentException] should be thrownBy Vector.unit[2](2)

  it should "have length 1" in:
    val unitVec = Vector.unit[3](0)
    unitVec.len should be (1f)

  it should "dot product with itself to give 1" in:
    0 until 3 foreach { i =>
      Vector.unit[3](i) * Vector.unit[3](i) should be (1f)
    }

  it should "dot product with unit vectors in other direction to give 0" in:
    0 until 3 foreach { i =>
      val otherIndex = (i + 1) % 3
      Vector.unit[3](i) * Vector.unit[3](otherIndex) should be (0f)
    }

  "float2String" should "format correctly in the default case (2 digits)" in:
    menger.common.float2string(1f) should be ("1")
    menger.common.float2string(1.1f) should be ("1.10")
    menger.common.float2string(1.123f) should be ("1.12")
    menger.common.float2string(1.126f) should be ("1.13")
    menger.common.float2string(1.0f) should be ("1")

  it should "work the same when 2 digits are explicitly specified" in:
    menger.common.float2String(2)(1f) should be ("1")
    menger.common.float2String(2)(1.1f) should be ("1.10")
    menger.common.float2String(2)(1.123f) should be ("1.12")
    menger.common.float2String(2)(1.126f) should be ("1.13")
    menger.common.float2String(2)(1.0f) should be ("1")

  it should "work correctly when 0 digits are specified" in:
    menger.common.float2String(0)(1f) should be ("1")
    menger.common.float2String(0)(1.1f) should be ("1")
    menger.common.float2String(0)(1.5f) should be ("2")
    menger.common.float2String(0)(1.9f) should be ("2")

  it should "work correctly when 3 digits are specified" in:
    menger.common.float2String(3)(1f) should be ("1")
    menger.common.float2String(3)(1.1f) should be ("1.100")
    menger.common.float2String(3)(1.1234f) should be ("1.123")
    menger.common.float2String(3)(1.1236f) should be ("1.124") // 1.1235 fails due to FP precision
    menger.common.float2String(3)(1.9999f) should be ("2.000")

  it should "work correctly with negative numbers" in:
    menger.common.float2String(2)(-1f) should be ("-1")
    menger.common.float2String(2)(-1.1f) should be ("-1.10")
    menger.common.float2String(2)(-1.123f) should be ("-1.12")
    menger.common.float2String(2)(-1.126f) should be ("-1.13")
