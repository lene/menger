package menger

import org.scalatest.funsuite.AnyFunSuite

class RotationProjectionParametersSuite extends AnyFunSuite {
  test("from CLI options") {
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
  }

  test("plus for rotation parameters") {
    val p1 = RotationProjectionParameters(1, 2, 3)
    val p2 = RotationProjectionParameters(4, 5, 6)
    val p3 = p1 + p2
    assert(p3.rotXW == 5)
    assert(p3.rotYW == 7)
    assert(p3.rotZW == 9)
  }

  test("plus for projection parameters - increase distance") {
    val originalEyeDistance = 2f
    val targetEyeDistance = 3f
    val p1 = RotationProjectionParameters(0, 0, 0, originalEyeDistance, 1)
    val p2 = RotationProjectionParameters(0, 0, 0, targetEyeDistance, 1)
    val p3 = p1 + p2
    assert(p3.eyeW > originalEyeDistance)
    assert(p3.eyeW < targetEyeDistance)
  }

  test("plus for projection parameters - decrease distance") {
    val originalEyeDistance = 3f
    val targetEyeDistance = 2f
    val p1 = RotationProjectionParameters(0, 0, 0, originalEyeDistance, 1)
    val p2 = RotationProjectionParameters(0, 0, 0, targetEyeDistance, 1)
    val p3 = p1 + p2
    assert(p3.eyeW < originalEyeDistance)
    assert(p3.eyeW > targetEyeDistance)
  }

  test("plus for projection parameters - keep distance same") {
    val originalEyeDistance = 3f
    val targetEyeDistance = 3f
    val p1 = RotationProjectionParameters(0, 0, 0, originalEyeDistance, 1)
    val p2 = RotationProjectionParameters(0, 0, 0, targetEyeDistance, 1)
    val p3 = p1 + p2
    assert(p3.eyeW == originalEyeDistance)
  }
}