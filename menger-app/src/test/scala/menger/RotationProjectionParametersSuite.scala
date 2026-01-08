package menger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class RotationProjectionParametersSuite extends AnyFlatSpec with Matchers:

  "instantiating from CLI options" should "work" in:
    val options = MengerCLIOptions(
      Seq(
        "--rot-x", "1", "--rot-y", "2", "--rot-z", "3",
        "--rot-x-w", "1", "--rot-y-w", "2", "--rot-z-w", "3",
        "--projection-screen-w", "1", "--projection-eye-w", "2"
      ))
    val parameters = RotationProjectionParameters(options)
    parameters.rotXW should be (1)
    parameters.rotYW should be (2)
    parameters.rotZW should be (3)
    parameters.rotX should be (1)
    parameters.rotY should be (2)
    parameters.rotZ should be (3)
    parameters.screenW should be (1)
    parameters.eyeW should be (2)
  
  "plus for rotation parameters" should "add the 4D components" in:
    val p1 = RotationProjectionParameters(1, 2, 3)
    val p2 = RotationProjectionParameters(4, 5, 6)
    val p3 = p1 + p2
    p3.rotXW should be (5)
    p3.rotYW should be (7)
    p3.rotZW should be (9)

  it should "add the 3D components" in:
    val p1 = RotationProjectionParameters(rotX = 1, rotY = 2, rotZ = 3)
    val p2 = RotationProjectionParameters(rotX = 4, rotY = 5, rotZ = 6)
    val p3 = p1 + p2
    p3.rotX should be (5)
    p3.rotY should be (7)
    p3.rotZ should be (9)

  "plus for projection parameters" should "increase distance if adding bigger distance" in:
    val originalEyeDistance = 2f
    val targetEyeDistance = 3f
    val p1 = RotationProjectionParameters(0, 0, 0, originalEyeDistance, 1)
    val p2 = RotationProjectionParameters(0, 0, 0, targetEyeDistance, 1)
    val p3 = p1 + p2
    p3.eyeW should be > originalEyeDistance
    p3.eyeW should be < targetEyeDistance

  it should "decrease distance if adding smaller distance" in:
    val originalEyeDistance = 3f
    val targetEyeDistance = 2f
    val p1 = RotationProjectionParameters(0, 0, 0, originalEyeDistance, 1)
    val p2 = RotationProjectionParameters(0, 0, 0, targetEyeDistance, 1)
    val p3 = p1 + p2
    p3.eyeW should be < originalEyeDistance
    p3.eyeW should be > targetEyeDistance

  it should "keep distance same if adding same distance" in:
    val originalEyeDistance = 3f
    val targetEyeDistance = 3f
    val p1 = RotationProjectionParameters(0, 0, 0, originalEyeDistance, 1)
    val p2 = RotationProjectionParameters(0, 0, 0, targetEyeDistance, 1)
    val p3 = p1 + p2
    p3.eyeW should be (originalEyeDistance)
