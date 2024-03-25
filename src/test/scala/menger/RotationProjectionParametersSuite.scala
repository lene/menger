package menger

import org.scalatest.flatspec.AnyFlatSpec

class RotationProjectionParametersSuite extends AnyFlatSpec:

  "instantiating from CLI options" should "work" in:
    val options = MengerCLIOptions(
      Seq(
        "--rot-x-w", "1", "--rot-y-w", "2", "--rot-z-w", "3",
        "--projection-screen-w", "1", "--projection-eye-w", "2"
      ))
    val parameters = RotationProjectionParameters(options)
    assert(parameters.rotXW == 1)
    assert(parameters.rotYW == 2)
    assert(parameters.rotZW == 3)
    assert(parameters.screenW == 1)
    assert(parameters.eyeW == 2)

  "plus for rotation parameters" should "add the components" in:
    val p1 = RotationProjectionParameters(1, 2, 3)
    val p2 = RotationProjectionParameters(4, 5, 6)
    val p3 = p1 + p2
    assert(p3.rotXW == 5)
    assert(p3.rotYW == 7)
    assert(p3.rotZW == 9)

  "plus for projection parameters" should "increase distance if adding bigger distance" in:
    val originalEyeDistance = 2f
    val targetEyeDistance = 3f
    val p1 = RotationProjectionParameters(0, 0, 0, originalEyeDistance, 1)
    val p2 = RotationProjectionParameters(0, 0, 0, targetEyeDistance, 1)
    val p3 = p1 + p2
    assert(p3.eyeW > originalEyeDistance)
    assert(p3.eyeW < targetEyeDistance)

  it should "decrease distance if adding smaller distance" in:
    val originalEyeDistance = 3f
    val targetEyeDistance = 2f
    val p1 = RotationProjectionParameters(0, 0, 0, originalEyeDistance, 1)
    val p2 = RotationProjectionParameters(0, 0, 0, targetEyeDistance, 1)
    val p3 = p1 + p2
    assert(p3.eyeW < originalEyeDistance)
    assert(p3.eyeW > targetEyeDistance)

  it should "keep distance same if adding same distance" in:
    val originalEyeDistance = 3f
    val targetEyeDistance = 3f
    val p1 = RotationProjectionParameters(0, 0, 0, originalEyeDistance, 1)
    val p2 = RotationProjectionParameters(0, 0, 0, targetEyeDistance, 1)
    val p3 = p1 + p2
    assert(p3.eyeW == originalEyeDistance)
