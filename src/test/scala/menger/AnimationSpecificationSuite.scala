package menger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AnimationSpecificationSuite extends AnyFlatSpec with Matchers:
  "RotationProjectionParameters for a frame" should "be correct if no rotation" in:
    AnimationSpecification("frames=10:rot-x-w=0-10").rotationProjectionParameters(0) shouldBe
      RotationProjectionParameters()

  it should "be correct for a single part with 3D rotation" in :
    AnimationSpecification("frames=10:rot-x=0-10").rotationProjectionParameters(5) shouldBe
      RotationProjectionParameters(rotX = 5)

  it should "be correct for a single part with 4D rotation" in:
    AnimationSpecification("frames=10:rot-x-w=0-10").rotationProjectionParameters(5) shouldBe
      RotationProjectionParameters(rotXW = 5)

  it should "be correct for a single part with multiple rotations" in:
    AnimationSpecification("frames=10:rot-x-w=0-10:rot-y-w=90-100").rotationProjectionParameters(5) shouldBe
      RotationProjectionParameters(rotXW = 5, rotYW = 95)

  it should "fail for a frame outside of the specified range" in:
    an [IllegalArgumentException] should be thrownBy
      AnimationSpecification("frames=10:rot-y-w=0-10").rotationProjectionParameters(20)
