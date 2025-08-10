package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.objects.Vector

class RectIndicesSuite extends AnyFlatSpec with Matchers:

  "A RectIndices" should "be able to be created with 4 indices" in:
    RectIndices(0, 1, 2, 3)

  it should "have a dimension of 4" in:
    RectIndices(0, 1, 2, 3).dimension should be(4)

  it should "not be able to be created with 3 indices" in:
    "RectIndices(0, 1, 2)" shouldNot typeCheck

  it should "not be able to be created with 5 indices" in:
    "RectIndices(0, 1, 2, 3, 4)" shouldNot typeCheck

  it should "correctly return the indices" in:
    RectIndices(0, 1, 2, 3).values should be(Seq(0, 1, 2, 3))

  it should "correctly return the indices by index" in:
    val indices = RectIndices(0, 1, 2, 3)
    indices(0) should be(0)
    indices(1) should be(1)
    indices(2) should be(2)
    indices(3) should be(3)

  it should "not be able to return an index out of bounds" in:
    val indices = RectIndices(0, 1, 2, 3)
    an [IllegalArgumentException] should be thrownBy indices(4)
    an [IllegalArgumentException] should be thrownBy indices(-1)

  it should "be able to convert to a Face4D" in:
    val indices = RectIndices(0, 1, 2, 3)
    val vertices = Seq(new Vector[4](0, 0, 0, 0), new Vector[4](0, 0, 0, 1), new Vector[4](0, 0, 1, 1), new Vector[4](0, 0, 1, 0))
    indices.toFace4D(vertices) should be(Face4D(vertices(0), vertices(1), vertices(2), vertices(3)))

  it should "not be able to convert to a Face4D if vertices do not match" in:
    val indices = RectIndices(0, 1, 2, 3)
    val vertices = Seq(new Vector[4](0, 0, 0, 0), new Vector[4](1, 1, 1, 1), new Vector[4](2, 2, 2, 2), new Vector[4](3, 3, 3, 3))
    an [IllegalArgumentException] should be thrownBy indices.toFace4D(vertices)

  "A sequence of RectIndices" should "be able to be created from tuples" in:
    val indices = RectIndices.fromTuples((0, 1, 2, 3), (4, 5, 6, 7))
    indices should have size 2
    indices(0) should be(RectIndices(0, 1, 2, 3))
    indices(1) should be(RectIndices(4, 5, 6, 7))

