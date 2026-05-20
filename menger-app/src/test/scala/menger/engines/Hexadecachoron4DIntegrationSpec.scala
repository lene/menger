package menger.engines

import menger.ObjectSpec
import menger.common.ObjectType
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class Hexadecachoron4DIntegrationSpec extends AnyFlatSpec with Matchers:

  "ObjectSpec parsing" should "parse hexadecachoron4d with level" in:
    val result = ObjectSpec.parse("type=hexadecachoron4d:level=3")
    result shouldBe a[Right[?, ?]]
    val spec = result.toOption.get
    spec.objectType shouldBe "hexadecachoron4d"
    spec.level shouldBe Some(3.0f)

  it should "parse hexadecachoron4d with 4D rotation params without error" in:
    // projection4D is populated by the scene builder at render time, not by ObjectSpec.parse;
    // we verify parse succeeds and the spec carries no projection4D (it uses defaults at build time)
    val result = ObjectSpec.parse("type=hexadecachoron4d:level=3:rot-xw=45:rot-yw=30:rot-zw=15")
    result shouldBe a[Right[?, ?]]
    result.toOption.get.projection4D shouldBe None

  it should "parse hexadecachoron4d with projection params without error" in:
    val result = ObjectSpec.parse("type=hexadecachoron4d:level=3:eye-w=4.0:screen-w=1.5")
    result shouldBe a[Right[?, ?]]
    result.toOption.get.projection4D shouldBe None

  it should "parse hexadecachoron4d with position and size" in:
    val result = ObjectSpec.parse("type=hexadecachoron4d:level=2:pos=1,2,3:size=2.0")
    result shouldBe a[Right[?, ?]]
    val spec = result.toOption.get
    spec.x shouldBe 1.0f
    spec.y shouldBe 2.0f
    spec.z shouldBe 3.0f
    spec.size shouldBe 2.0f

  it should "parse hexadecachoron4d with material" in:
    val result = ObjectSpec.parse("type=hexadecachoron4d:level=3:material=chrome")
    result shouldBe a[Right[?, ?]]
    result.toOption.get.material shouldBe defined

  it should "require level for hexadecachoron4d" in:
    val result = ObjectSpec.parse("type=hexadecachoron4d")
    result shouldBe a[Left[?, ?]]
    result.left.getOrElse("") should include("level")

  it should "reject negative level for hexadecachoron4d" in:
    val result = ObjectSpec.parse("type=hexadecachoron4d:level=-1")
    result shouldBe a[Left[?, ?]]

  it should "accept level 0 for hexadecachoron4d" in:
    val result = ObjectSpec.parse("type=hexadecachoron4d:level=0")
    result shouldBe a[Right[?, ?]]

  it should "accept fractional level at parse time (rejection is in SceneBuilder.validate)" in:
    // ObjectSpec.parse does not reject fractional levels; Hexadecachoron4DSceneBuilder.validate does
    val result = ObjectSpec.parse("type=hexadecachoron4d:level=1.5")
    result shouldBe a[Right[?, ?]]

  it should "not have distanceThreshold for hexadecachoron4d" in:
    val result = ObjectSpec.parse("type=hexadecachoron4d:level=3")
    result shouldBe a[Right[?, ?]]
    result.toOption.get.distanceThreshold shouldBe None

  "Type classification" should "identify hexadecachoron4d as hexadecachoron4d" in:
    ObjectType.isHexadecachoron4D("hexadecachoron4d") shouldBe true

  it should "identify hexadecachoron4d as 4D sponge" in:
    ObjectType.is4DSponge("hexadecachoron4d") shouldBe true

  it should "not classify hexadecachoron4d as projected-4D mesh" in:
    ObjectType.isProjected4D("hexadecachoron4d") shouldBe false

  it should "not classify hexadecachoron4d as triangle mesh" in:
    ObjectType.isTriangleMesh("hexadecachoron4d") shouldBe false

  it should "not classify hexadecachoron4d as 3D sponge" in:
    ObjectType.isSponge("hexadecachoron4d") shouldBe false
