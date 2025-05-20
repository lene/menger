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

  "isRotationAxisSet" should "be true for X if X is set in AnimationSpecification" in:
    AnimationSpecification("frames=10:rot-x=0-10").isRotationAxisSet(
      x = 1, y = 0, z = 0, xw = 0, yw = 0, zw = 0
    ) shouldBe true

  it should "be true for Y if Y is set in AnimationSpecification" in:
    AnimationSpecification("frames=10:rot-y=0-10").isRotationAxisSet(
      x = 0, y = 1, z = 0, xw = 0, yw = 0, zw = 0
    ) shouldBe true

  it should "be true for Z if Z is set in AnimationSpecification" in:
    AnimationSpecification("frames=10:rot-z=0-10").isRotationAxisSet(
      x = 0, y = 0, z = 1, xw = 0, yw = 0, zw = 0
    ) shouldBe true

  it should "be true for XW if XW is set in AnimationSpecification" in:
    AnimationSpecification("frames=10:rot-x-w=0-10").isRotationAxisSet(
      x = 0, y = 0, z = 0, xw = 1, yw = 0, zw = 0
    ) shouldBe true

  it should "be true for YW if YW is set in AnimationSpecification" in:
    AnimationSpecification("frames=10:rot-y-w=0-10").isRotationAxisSet(
      x = 0, y = 0, z = 0, xw = 0, yw = 1, zw = 0
    ) shouldBe true

  it should "be true for ZW if ZW is set in AnimationSpecification" in:
    AnimationSpecification("frames=10:rot-z-w=0-10").isRotationAxisSet(
      x = 0, y = 0, z = 0, xw = 0, yw = 0, zw = 1
    ) shouldBe true

  it should "be false if x axis is set but not in AnimationSpecification" in:
    AnimationSpecification("frames=10:rot-y=0-10").isRotationAxisSet(
      x = 1, y = 0, z = 0, xw = 0, yw = 0, zw = 0
    ) shouldBe false

  it should "be false if y axis is set but not in AnimationSpecification" in:
    AnimationSpecification("frames=10:rot-x=0-10").isRotationAxisSet(
      x = 0, y = 1, z = 0, xw = 0, yw = 0, zw = 0
    ) shouldBe false

  it should "be false if z axis is set but not in AnimationSpecification" in:
    AnimationSpecification("frames=10:rot-x=0-10").isRotationAxisSet(
      x = 0, y = 0, z = 1, xw = 0, yw = 0, zw = 0
    ) shouldBe false

  it should "be false if xw axis is set but not in AnimationSpecification" in:
    AnimationSpecification("frames=10:rot-y=0-10").isRotationAxisSet(
      x = 0, y = 0, z = 0, xw = 1, yw = 0, zw = 0
    ) shouldBe false

  it should "be false if yw axis is set but not in AnimationSpecification" in:
    AnimationSpecification("frames=10:rot-x=0-10").isRotationAxisSet(
      x = 0, y = 0, z = 0, xw = 0, yw = 1, zw = 0
    ) shouldBe false

  it should "be false if zw axis is set but not in AnimationSpecification" in:
    AnimationSpecification("frames=10:rot-x=0-10").isRotationAxisSet(
      x = 0, y = 0, z = 0, xw = 0, yw = 0, zw = 1
    ) shouldBe false