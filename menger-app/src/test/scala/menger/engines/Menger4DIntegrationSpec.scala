package menger.engines

import menger.ObjectSpec
import menger.common.ObjectType
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class Menger4DIntegrationSpec extends AnyFlatSpec with Matchers:

  "ObjectSpec parsing" should "parse menger4d with level" in:
    val result = ObjectSpec.parse("type=menger4d:level=5")
    result shouldBe a[Right[?, ?]]
    val spec = result.toOption.get
    spec.objectType shouldBe "menger4d"
    spec.level shouldBe Some(5.0f)

  it should "parse menger4d with 4D rotation" in:
    val result = ObjectSpec.parse("type=menger4d:level=3:rot-xw=45:rot-yw=30:rot-zw=15")
    result shouldBe a[Right[?, ?]]
    val proj = result.toOption.get.projection4D.get
    proj.rotXW shouldBe 45f
    proj.rotYW shouldBe 30f
    proj.rotZW shouldBe 15f

  it should "parse menger4d with projection parameters" in:
    val result = ObjectSpec.parse("type=menger4d:level=3:eye-w=4.0:screen-w=1.5")
    result shouldBe a[Right[?, ?]]
    val proj = result.toOption.get.projection4D.get
    proj.eyeW shouldBe 4.0f
    proj.screenW shouldBe 1.5f

  it should "parse menger4d with position and size" in:
    val result = ObjectSpec.parse("type=menger4d:level=2:pos=1,2,3:size=2.0")
    result shouldBe a[Right[?, ?]]
    val spec = result.toOption.get
    spec.x shouldBe 1.0f
    spec.y shouldBe 2.0f
    spec.z shouldBe 3.0f
    spec.size shouldBe 2.0f

  it should "parse menger4d with material" in:
    val result = ObjectSpec.parse("type=menger4d:level=3:material=chrome")
    result shouldBe a[Right[?, ?]]
    result.toOption.get.material shouldBe defined

  it should "require level for menger4d" in:
    val result = ObjectSpec.parse("type=menger4d")
    result shouldBe a[Left[?, ?]]
    result.left.getOrElse("") should include("level")

  it should "reject negative level for menger4d" in:
    val result = ObjectSpec.parse("type=menger4d:level=-1")
    result shouldBe a[Left[?, ?]]

  it should "accept level 0 for menger4d" in:
    val result = ObjectSpec.parse("type=menger4d:level=0")
    result shouldBe a[Right[?, ?]]

  it should "parse menger4d with distanceThreshold" in:
    val result = ObjectSpec.parse("type=menger4d:level=3:dist-threshold=3")
    result shouldBe a[Right[?, ?]]
    result.toOption.get.distanceThreshold shouldBe Some(3)

  it should "default distanceThreshold to None when not specified" in:
    val result = ObjectSpec.parse("type=menger4d:level=2")
    result shouldBe a[Right[?, ?]]
    result.toOption.get.distanceThreshold shouldBe None

  "Type classification" should "identify menger4d as valid" in:
    ObjectType.isValid("menger4d") shouldBe true

  it should "identify menger4d as 4D sponge" in:
    ObjectType.is4DSponge("menger4d") shouldBe true

  it should "not classify menger4d as projected-4D mesh" in:
    ObjectType.isProjected4D("menger4d") shouldBe false

  it should "not classify menger4d as triangle mesh" in:
    ObjectType.isTriangleMesh("menger4d") shouldBe false

  it should "not classify menger4d as 3D sponge" in:
    ObjectType.isSponge("menger4d") shouldBe false
