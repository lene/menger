package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class RotationProjectionParametersSuite extends AnyFlatSpec with Matchers:

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
    val p1 = RotationProjectionParameters(0, 0, 0, Some(originalEyeDistance), 1)
    val p2 = RotationProjectionParameters(0, 0, 0, Some(targetEyeDistance), 1)
    val p3 = p1 + p2
    p3.eyeW.get should be > originalEyeDistance
    p3.eyeW.get should be < targetEyeDistance

  it should "decrease distance if adding smaller distance" in:
    val originalEyeDistance = 3f
    val targetEyeDistance = 2f
    val p1 = RotationProjectionParameters(0, 0, 0, Some(originalEyeDistance), 1)
    val p2 = RotationProjectionParameters(0, 0, 0, Some(targetEyeDistance), 1)
    val p3 = p1 + p2
    p3.eyeW.get should be < originalEyeDistance
    p3.eyeW.get should be > targetEyeDistance

  it should "keep distance same if adding same distance" in:
    val originalEyeDistance = 3f
    val targetEyeDistance = 3f
    val p1 = RotationProjectionParameters(0, 0, 0, Some(originalEyeDistance), 1)
    val p2 = RotationProjectionParameters(0, 0, 0, Some(targetEyeDistance), 1)
    val p3 = p1 + p2
    p3.eyeW should be (Some(originalEyeDistance))

  it should "have no eyeW after adding two rotation-only changes" in:
    val sum = RotationProjectionParameters(rotXW = 1f) + RotationProjectionParameters(rotYW = 2f)
    sum.eyeW shouldBe None

  it should "keep an eyeW set on only one side" in:
    val rotationOnly = RotationProjectionParameters(rotXW = 1f)
    val scrolled = RotationProjectionParameters(eyeW = Some(4f))
    (rotationOnly + scrolled).eyeW shouldBe defined
    (scrolled + rotationOnly).eyeW shouldBe defined

  it should "treat the default eyeW as a real value, not as 'unchanged'" in:
    val explicit = RotationProjectionParameters(eyeW = Some(menger.common.Const.defaultEyeW))
    explicit.eyeW shouldBe defined
    explicit should not be RotationProjectionParameters()

  "RotationProjectionParameters.toString" should "include XYZ rotation when non-zero" in:
    val p = RotationProjectionParameters(rotX = 1.0f, rotY = 0.5f, rotZ = -0.5f)
    p.toString should include ("x: 1.0")
    p.toString should include ("y: 0.5")
    p.toString should include ("z: -0.5")

  it should "include 4D rotation when non-zero" in:
    val p = RotationProjectionParameters(rotXW = 1.0f, rotYW = 2.0f, rotZW = 0.5f)
    p.toString should include ("xw: 1.0")
    p.toString should include ("yw: 2.0")
    p.toString should include ("zw: 0.5")

  it should "include eyeW when set" in:
    val p = RotationProjectionParameters(eyeW = Some(5.0f))
    p.toString should include ("eyeW=5.0")

  it should "not mention eyeW when it is not set" in:
    RotationProjectionParameters(rotXW = 1f).toString should not include "eyeW"

  it should "include screenW when non-default" in:
    val p = RotationProjectionParameters(screenW = 3.0f)
    p.toString should include ("screenW=3.0")

  it should "not include zero components" in:
    val p = RotationProjectionParameters()
    p.toString should not include "x:"
    p.toString should not include "xw:"
    p.toString should not include "eyeW="
    p.toString should not include "screenW="
