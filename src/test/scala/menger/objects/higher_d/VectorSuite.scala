package menger.objects.higher_d

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
