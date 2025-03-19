package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class QuadInfoSuite extends AnyFlatSpec with Matchers:
  "A RectInfo" should "be able to be created with 4 VertexInfos" in:
    QuadInfo(VertexInfo(0, 0, 0), VertexInfo(1, 0, 0), VertexInfo(1, 1, 0), VertexInfo(0, 1, 0))

  it should "have a dimension of 4" in:
    QuadInfo(VertexInfo(0, 0, 0), VertexInfo(1, 0, 0), VertexInfo(1, 1, 0), VertexInfo(0, 1, 0)).dimension should be (4)

  it should "not be able to be created with 3 VertexInfos" in:
    "QuadInfo(VertexInfo(0, 0, 0), VertexInfo(1, 0, 0), VertexInfo(1, 1, 0))" shouldNot typeCheck

  it should "not be able to be created with 5 VertexInfos" in:
    "QuadInfo(VertexInfo(0, 0, 0), VertexInfo(1, 0, 0), VertexInfo(1, 1, 0), VertexInfo(0, 1, 0), VertexInfo(0, 0, 0))" shouldNot typeCheck
