package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class FixedVectorSuite extends AnyFlatSpec with Matchers:

  "A FixedVector size 3" should "be able to be created with 3 elements" in:
    FixedVector[3, Int](1, 2, 3)

  it should "have a dimension of 3" in:
    FixedVector[3, Int](1, 2, 3).dimension should be(3)

  it should "not be able to be created with 2 elements" in:
    an [IllegalArgumentException] should be thrownBy FixedVector[3, Int](1, 2)

  it should "not be able to be created with 4 elements" in:
    an [IllegalArgumentException] should be thrownBy FixedVector[3, Int](1, 2, 3, 4)

  it should "correctly return the elements" in:
    FixedVector[3, Int](1, 2, 3).values should be(Seq(1, 2, 3))

  it should "correctly return the elements by index" in:
    val vec = FixedVector[3, Int](1, 2, 3)
    vec(0) should be(1)
    vec(1) should be(2)
    vec(2) should be(3)

  it should "not be able to return an element out of bounds" in:
    val vec = FixedVector[3, Int](1, 2, 3)
    an [IllegalArgumentException] should be thrownBy vec(3)
    an [IllegalArgumentException] should be thrownBy vec(-1)

  "A FixedVector size 4" should "be able to be created with 4 elements" in:
    FixedVector[4, Int](1, 2, 3, 4)

  "A FixedVector size 5" should "be able to be created with 5 elements" in:
    FixedVector[5, Int](1, 2, 3, 4, 5)

