package menger.spec

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ObjectSpecScaleSuite extends AnyFlatSpec with Matchers:

  "ObjectSpec.nonIdentityScale" should "treat an all-ones scale as no scale" in:
    ObjectSpec.nonIdentityScale(IndexedSeq(1f, 1f, 1f, 1f)) shouldBe None

  it should "keep any other scale as it is" in:
    ObjectSpec.nonIdentityScale(IndexedSeq(1f, 1f, 1f, 0.5f)) shouldBe
      Some(IndexedSeq(1f, 1f, 1f, 0.5f))
    ObjectSpec.nonIdentityScale(IndexedSeq(2f, 1f, 1f, 1f, 1f)) shouldBe
      Some(IndexedSeq(2f, 1f, 1f, 1f, 1f))
