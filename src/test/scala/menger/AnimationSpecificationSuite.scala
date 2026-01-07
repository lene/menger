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

  "AnimationSpecification" should "be invalid when specifying seconds as time metric" in :
    AnimationSpecification("seconds=10:rot-x-w=0-10").isTimeSpecValid shouldBe false

  it should "be invalid when specifying no time metric" in :
    AnimationSpecification("rot-x-w=0-10").isTimeSpecValid shouldBe false

  // Edge case tests for frames parameter
  "frames parameter edge cases" should "be invalid for frames=0" in:
    AnimationSpecification("frames=0:rot-x=0-10").isTimeSpecValid shouldBe false

  it should "be invalid for negative frames" in:
    AnimationSpecification("frames=-1:rot-x=0-10").isTimeSpecValid shouldBe false
    AnimationSpecification("frames=-100:rot-x=0-10").isTimeSpecValid shouldBe false

  it should "be invalid for non-integer frames" in:
    AnimationSpecification("frames=1.5:rot-x=0-10").isTimeSpecValid shouldBe false

  it should "be invalid for non-numeric frames" in:
    AnimationSpecification("frames=abc:rot-x=0-10").isTimeSpecValid shouldBe false

  it should "handle large frame counts" in:
    val spec = AnimationSpecification("frames=1000000:rot-x=0-10")
    spec.isTimeSpecValid shouldBe true
    spec.frames shouldBe Some(1000000)

  // Edge case tests for parsing
  "parsing edge cases" should "handle empty string gracefully" in:
    val spec = AnimationSpecification("")
    spec.isTimeSpecValid shouldBe false
    spec.animationParameters shouldBe empty

  it should "handle string with only delimiters" in:
    val spec = AnimationSpecification(":::")
    spec.isTimeSpecValid shouldBe false

  it should "handle string with only equals signs" in:
    val spec = AnimationSpecification("===")
    spec.isTimeSpecValid shouldBe false

  it should "handle malformed key-value pairs" in:
    val spec = AnimationSpecification("frames10:rot-x=0-10")
    spec.isTimeSpecValid shouldBe false  // missing = in frames

  // Edge case tests for range parsing
  // Note: The current parser uses '-' as delimiter, so negative values are not supported
  // This is a known limitation - negative angles would need a different delimiter or escaping
  "range parsing edge cases" should "fail for negative start value (parser limitation)" in:
    an[IllegalArgumentException] should be thrownBy:
      AnimationSpecification("frames=10:rot-x=-90-0").animationParameters

  it should "fail for negative end value (parser limitation)" in:
    an[IllegalArgumentException] should be thrownBy:
      AnimationSpecification("frames=10:rot-x=0--90").animationParameters

  it should "fail for both negative values (parser limitation)" in:
    an[IllegalArgumentException] should be thrownBy:
      AnimationSpecification("frames=10:rot-x=-90--45").animationParameters

  it should "handle floating point values in range" in:
    val spec = AnimationSpecification("frames=10:level=0.5-2.5")
    spec.animationParameters("level") shouldBe (0.5f, 2.5f)

  it should "fail for malformed range with missing end" in:
    an[IllegalArgumentException] should be thrownBy:
      AnimationSpecification("frames=10:rot-x=0-").animationParameters

  it should "fail for malformed range with missing start" in:
    an[IllegalArgumentException] should be thrownBy:
      AnimationSpecification("frames=10:rot-x=-10").animationParameters

  it should "fail for range with non-numeric values" in:
    an[IllegalArgumentException] should be thrownBy:
      AnimationSpecification("frames=10:rot-x=start-end").animationParameters

  // Edge case tests for interpolation (current method)
  "interpolation edge cases" should "return start value at frame 0" in:
    val spec = AnimationSpecification("frames=10:rot-x=100-200")
    spec.rotationProjectionParameters(0).rotX shouldBe 100f

  it should "return end value at last frame" in:
    val spec = AnimationSpecification("frames=10:rot-x=100-200")
    spec.rotationProjectionParameters(10).rotX shouldBe 200f

  it should "interpolate correctly at midpoint" in:
    val spec = AnimationSpecification("frames=10:rot-x=0-100")
    spec.rotationProjectionParameters(5).rotX shouldBe 50f

  it should "handle equal start and end values" in:
    val spec = AnimationSpecification("frames=10:rot-x=45-45")
    spec.rotationProjectionParameters(0).rotX shouldBe 45f
    spec.rotationProjectionParameters(5).rotX shouldBe 45f
    spec.rotationProjectionParameters(10).rotX shouldBe 45f

  it should "handle reverse animation (end < start)" in:
    val spec = AnimationSpecification("frames=10:rot-x=100-0")
    spec.rotationProjectionParameters(0).rotX shouldBe 100f
    spec.rotationProjectionParameters(5).rotX shouldBe 50f
    spec.rotationProjectionParameters(10).rotX shouldBe 0f

  // Edge case tests for level() method
  "level method edge cases" should "return None when level not specified" in:
    val spec = AnimationSpecification("frames=10:rot-x=0-10")
    spec.level(5) shouldBe None

  it should "return correct value when level is specified" in:
    val spec = AnimationSpecification("frames=10:level=0-5")
    spec.level(0) shouldBe Some(0f)
    spec.level(5) shouldBe Some(2.5f)
    spec.level(10) shouldBe Some(5f)

  // Edge case tests for frames=1 (single frame animation)
  "single frame animation" should "work with frames=1" in:
    val spec = AnimationSpecification("frames=1:rot-x=0-360")
    spec.isTimeSpecValid shouldBe true
    spec.rotationProjectionParameters(0).rotX shouldBe 0f
    spec.rotationProjectionParameters(1).rotX shouldBe 360f
