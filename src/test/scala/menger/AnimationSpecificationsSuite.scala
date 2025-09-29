package menger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AnimationSpecificationsSuite extends AnyFlatSpec with Matchers:
  "AnimationSpecifications" should "be empty by default" in:
    AnimationSpecifications().parts should be (empty)

  "Finding the animation specification for a frame" should "return the frame if only one part" in:
    AnimationSpecifications(List("frames=10:rot-x-w=0-10")).partAndFrame(5).get shouldBe (
      List(AnimationSpecification("frames=10:rot-x-w=0-10")), 5
    )

  it should "return the correct part if only one part and multiple rotations" in:
    AnimationSpecifications(
      List("frames=10:rot-x-w=0-10:rot-y-w=90-100")
    ).partAndFrame(5).get shouldBe (
      List(AnimationSpecification("frames=10:rot-x-w=0-10:rot-y-w=90-100")), 5
    )


  it should "return the correct parts and frame for multiple parts" in:
    AnimationSpecifications(List(
      "frames=10:rot-x-w=0-10", "frames=10:rot-y-w=0-10")
    ).partAndFrame(15).get shouldBe (
      List(AnimationSpecification("frames=10:rot-x-w=0-10"), AnimationSpecification("frames=10:rot-y-w=0-10")), 5
    )

  it should "only return the relevant parts for multiple parts if frame is in the first part" in:
    AnimationSpecifications(
      List("frames=10:rot-x-w=0-10", "frames=10:rot-y-w=0-10")
    ).partAndFrame(5).get shouldBe (
      List(AnimationSpecification("frames=10:rot-x-w=0-10")), 5
    )

  it should "fail for a frame outside of the specified range" in:
    AnimationSpecifications(
      List("frames=10:rot-x-w=0-10", "frames=10:rot-y-w=0-10")
    ).partAndFrame(20).isFailure should be(true)

  "RotationProjectionParameters for a frame" should "be correct if no rotation" in:
    AnimationSpecifications(
      List("frames=10:rot-x-w=0-10")
    ).rotationProjectionParameters(0).get shouldBe RotationProjectionParameters()

  it should "be correct for a single part" in:
    AnimationSpecifications(
      List("frames=10:rot-x-w=0-10")
    ).rotationProjectionParameters(5).get shouldBe RotationProjectionParameters(
      rotXW = 5
    )

  it should "be correct for a single part with multiple rotations" in:
    AnimationSpecifications(
      List("frames=10:rot-x-w=0-10:rot-y-w=90-100")
    ).rotationProjectionParameters(5).get shouldBe RotationProjectionParameters(
      rotXW = 5, rotYW = 95
    )

  it should "be correct for a single part with 3D rotation" in :
    AnimationSpecifications(
      List("frames=10:rot-x=0-10")
    ).rotationProjectionParameters(5).get shouldBe RotationProjectionParameters(
      rotX = 5
    )

  it should "take the first rotation into account when there are two parts" in:
    AnimationSpecifications(
      List("frames=10:rot-x-w=0-10", "frames=10:rot-y-w=0-10")
    ).rotationProjectionParameters(15).get shouldBe RotationProjectionParameters(
      rotXW = 10, rotYW = 5
    )

  it should "also work when there are three parts" in:
    AnimationSpecifications(
      List("frames=10:rot-x-w=0-10", "frames=10:rot-y-w=0-10", "frames=10:rot-z-w=0-10")
    ).rotationProjectionParameters(25).get shouldBe RotationProjectionParameters(
      rotXW = 10, rotYW = 10, rotZW = 5
    )

  it should "fail for a frame outside of the specified range" in:
    AnimationSpecifications(
      List("frames=10:rot-x-w=0-10", "frames=10:rot-y-w=0-10")
    ).rotationProjectionParameters(20).isFailure should be(true)

  it should "fail when specifying seconds as time metric" in:
    an [IllegalArgumentException] should be thrownBy
      AnimationSpecifications(List("seconds=10:rot-x-w=0-10"))
