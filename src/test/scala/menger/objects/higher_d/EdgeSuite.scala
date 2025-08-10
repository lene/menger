package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.objects.Vector

class EdgeSuite extends AnyFlatSpec with Matchers:
  "An Edge" should "be able to be created with 2 Vector4s" in:
    Edge(Vector[4](0, 0, 0, 0), Vector[4](1, 0, 0, 0))

  it should "have a dimension of 2" in:
    Edge(Vector[4](0, 0, 0, 0), Vector[4](1, 0, 0, 0)).dimension should be(2)

  it should "not be able to be created with 3 Vector4s" in:
    "Edge(Vector[4](0, 0, 0, 0), Vector[4](1, 0, 0, 0), Vector[4](1, 1, 0, 0))" shouldNot typeCheck

  it should "not be able to be created with 1 Vector4" in:
    "Edge(Vector[4](0, 0, 0, 0))" shouldNot typeCheck

  it should "correctly return the vertices" in:
    Edge(Vector[4](0, 0, 0, 0), Vector[4](1, 0, 0, 0)).values should be(Seq(Vector[4](0, 0, 0, 0), Vector[4](1, 0, 0, 0)))

  it should "correctly return the vertices by index" in:
    val edge = Edge(Vector[4](0, 0, 0, 0), Vector[4](1, 0, 0, 0))
    edge(0) should be(Vector[4](0, 0, 0, 0))
    edge(1) should be(Vector[4](1, 0, 0, 0))

  it should "not be able to return a vertex out of bounds" in:
    val edge = Edge(Vector[4](0, 0, 0, 0), Vector[4](1, 0, 0, 0))
    an [IllegalArgumentException] should be thrownBy edge(2)
    an [IllegalArgumentException] should be thrownBy edge(-1)

  it should "calculate the diff between the vertices" in:
    Edge(Vector[4](0, 0, 0, 0), Vector[4](1, 0, 0, 0)).diff should be (Vector[4](1, 0, 0, 0))

  it should "respect the order of the edges when calculating the diff" in:
    Edge(Vector[4](1, 0, 0, 0), Vector[4](0, 0, 0, 0)).diff should be (Vector[4](-1, 0, 0, 0))

  "an Edge's string representation" should "be correct" in :
    val edge = Edge(Vector[4](0, 0, 0, 0), Vector[4](1, 1, 1, 1))
    edge.toString should include("<0, 0, 0, 0>")
    edge.toString should include("<1, 1, 1, 1>")

