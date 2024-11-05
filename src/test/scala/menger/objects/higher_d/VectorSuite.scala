package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec

class VectorSuite extends AnyFlatSpec:
  "A Vector of any dimension" should "be instatiable" in:
    val v1 = Vector[1](1)
    val v2 = Vector[2](1, 2)

  it should "have the correct number of elements" in:
    val v1 = Vector[1](1)
    assert(v1.dimension == 1)
    assert(v1.v.size == 1)

  it should "fail when instantiated with the wrong number of elements" in:
    assertThrows[AssertionError](Vector[1](1, 2))
    assertThrows[AssertionError](Vector[2](1))
