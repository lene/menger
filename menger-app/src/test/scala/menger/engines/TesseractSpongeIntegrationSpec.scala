package menger.engines

import menger.ObjectSpec
import menger.common.ObjectType
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TesseractSpongeIntegrationSpec extends AnyFlatSpec with Matchers:

  // === ObjectSpec Parsing Tests ===

  "ObjectSpec parsing" should "parse tesseract-sponge with level" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.objectType shouldBe "tesseract-sponge"
    spec.level shouldBe Some(1.0f)

  it should "parse tesseract-sponge-2 with level" in:
    val result = ObjectSpec.parse("type=tesseract-sponge-2:level=2")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.objectType shouldBe "tesseract-sponge-2"
    spec.level shouldBe Some(2.0f)

  it should "parse tesseract-sponge with fractional level" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1.5")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.level shouldBe Some(1.5f)

  it should "parse tesseract-sponge with position and size" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1:pos=1,2,3:size=2.5")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.x shouldBe 1.0f
    spec.y shouldBe 2.0f
    spec.z shouldBe 3.0f
    spec.size shouldBe 2.5f

  it should "parse tesseract-sponge with 4D rotation" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1:rot-xw=45:rot-yw=30:rot-zw=15")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    val proj = spec.projection4D.get
    proj.rotXW shouldBe 45f
    proj.rotYW shouldBe 30f
    proj.rotZW shouldBe 15f

  it should "parse tesseract-sponge with 4D projection parameters" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1:eye-w=5.0:screen-w=2.5")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    val proj = spec.projection4D.get
    proj.eyeW shouldBe 5.0f
    proj.screenW shouldBe 2.5f

  // === Level Validation Tests ===

  "Level validation" should "require level for tesseract-sponge" in:
    val result = ObjectSpec.parse("type=tesseract-sponge")
    result shouldBe a[Left[?, ?]]
    result.left.getOrElse("") should include("level")

  it should "require level for tesseract-sponge-2" in:
    val result = ObjectSpec.parse("type=tesseract-sponge-2")
    result shouldBe a[Left[?, ?]]
    result.left.getOrElse("") should include("level")

  it should "reject negative levels" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=-1")
    result shouldBe a[Left[?, ?]]
    result.left.getOrElse("") should include("non-negative")

  it should "accept level 0" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=0")
    result shouldBe a[Right[?, ?]]

  // === Type Classification Tests ===

  "Type classification" should "identify tesseract-sponge as hypercube" in:
    ObjectType.isProjected4D("tesseract-sponge") shouldBe true
    ObjectType.isProjected4D("TESSERACT-SPONGE") shouldBe true

  it should "identify tesseract-sponge-2 as hypercube" in:
    ObjectType.isProjected4D("tesseract-sponge-2") shouldBe true
    ObjectType.isProjected4D("TESSERACT-SPONGE-2") shouldBe true

  it should "identify tesseract-sponge as 4D sponge" in:
    ObjectType.is4DSponge("tesseract-sponge") shouldBe true
    ObjectType.is4DSponge("TESSERACT-SPONGE") shouldBe true

  it should "identify tesseract-sponge-2 as 4D sponge" in:
    ObjectType.is4DSponge("tesseract-sponge-2") shouldBe true
    ObjectType.is4DSponge("TESSERACT-SPONGE-2") shouldBe true

  it should "not classify tesseract-sponge as 3D sponge" in:
    ObjectType.isSponge("tesseract-sponge") shouldBe false
    ObjectType.isSponge("tesseract-sponge-2") shouldBe false

  it should "classify both sponge types as valid" in:
    ObjectType.isValid("tesseract-sponge") shouldBe true
    ObjectType.isValid("tesseract-sponge-2") shouldBe true

  // === Material Support Tests ===

  "Material support" should "parse tesseract-sponge with glass material" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1:material=glass")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.material shouldBe defined
    spec.material.get.ior shouldBe 1.5f

  it should "parse tesseract-sponge with chrome material" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1:material=chrome")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.material shouldBe defined
    spec.material.get.metallic shouldBe 1.0f

  it should "parse tesseract-sponge with color override" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1:color=#FF5500")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.color shouldBe defined
    spec.color.get.r shouldBe 1.0f
    spec.color.get.g should be(0.33f +- 0.01f)
    spec.color.get.b shouldBe 0.0f

  it should "parse tesseract-sponge with material and IOR override" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1:material=glass:ior=2.0")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.material shouldBe defined
    spec.material.get.ior shouldBe 2.0f

  // === Edge Rendering Tests ===

  "Edge rendering" should "parse tesseract-sponge with edge-radius" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1:edge-radius=0.03")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.edgeRadius shouldBe Some(0.03f)

  it should "parse tesseract-sponge with edge-material" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1:edge-material=film")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.edgeMaterial shouldBe defined
    spec.hasEdgeRendering shouldBe true

  it should "parse tesseract-sponge with both face and edge materials" in:
    val result = ObjectSpec.parse(
      "type=tesseract-sponge:level=1:material=glass:edge-material=chrome:edge-radius=0.015"
    )
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.material shouldBe defined
    spec.edgeMaterial shouldBe defined
    spec.edgeRadius shouldBe Some(0.015f)

  it should "parse tesseract-sponge with edge-color and edge-emission" in:
    val result = ObjectSpec.parse(
      "type=tesseract-sponge:level=1:edge-color=#00FFFF:edge-emission=5.0"
    )
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.edgeMaterial shouldBe defined
    spec.hasEdgeRendering shouldBe true

  // === Multiple Objects Tests ===

  "Multiple objects" should "parse list of tesseract-sponge specs" in:
    val specs = List(
      "type=tesseract-sponge:level=1:pos=-2,0,0:color=#FF0000",
      "type=tesseract-sponge:level=1:pos=2,0,0:color=#00FF00"
    )
    val result = ObjectSpec.parseAll(specs)
    result shouldBe a[Right[?, ?]]

    val objects = result.toOption.get
    objects.length shouldBe 2
    objects(0).objectType shouldBe "tesseract-sponge"
    objects(1).objectType shouldBe "tesseract-sponge"
    objects(0).x shouldBe -2.0f
    objects(1).x shouldBe 2.0f

  it should "parse mixed tesseract and tesseract-sponge specs" in:
    val specs = List(
      "type=tesseract:pos=-2,0,0",
      "type=tesseract-sponge:level=1:pos=2,0,0"
    )
    val result = ObjectSpec.parseAll(specs)
    result shouldBe a[Right[?, ?]]

    val objects = result.toOption.get
    objects.length shouldBe 2
    objects(0).objectType shouldBe "tesseract"
    objects(1).objectType shouldBe "tesseract-sponge"

  it should "parse both sponge types together" in:
    val specs = List(
      "type=tesseract-sponge:level=1:pos=-2,0,0",
      "type=tesseract-sponge-2:level=2:pos=2,0,0"
    )
    val result = ObjectSpec.parseAll(specs)
    result shouldBe a[Right[?, ?]]

    val objects = result.toOption.get
    objects(0).objectType shouldBe "tesseract-sponge"
    objects(1).objectType shouldBe "tesseract-sponge-2"

  // === 4D Projection Compatibility Tests ===

  "4D projection compatibility" should "identify same rotation as compatible" in:
    val spec1 = ObjectSpec.parse("type=tesseract-sponge:level=1:rot-xw=30:rot-yw=20")
    val spec2 = ObjectSpec.parse("type=tesseract-sponge:level=1:rot-xw=30:rot-yw=20")

    spec1 shouldBe a[Right[?, ?]]
    spec2 shouldBe a[Right[?, ?]]

    val s1 = spec1.toOption.get
    val s2 = spec2.toOption.get

    s1.projection4D.get shouldBe s2.projection4D.get

  it should "identify different rotation as incompatible" in:
    val spec1 = ObjectSpec.parse("type=tesseract-sponge:level=1:rot-xw=30")
    val spec2 = ObjectSpec.parse("type=tesseract-sponge:level=1:rot-xw=45")

    val s1 = spec1.toOption.get
    val s2 = spec2.toOption.get

    s1.projection4D.get should not be s2.projection4D.get

  // === Combined Parameters Tests ===

  "Combined parameters" should "parse tesseract-sponge with all parameters" in:
    val result = ObjectSpec.parse(
      "type=tesseract-sponge:level=1:pos=1,2,3:size=2.5:color=#4488FF:" +
      "rot-xw=30:rot-yw=20:rot-zw=10:eye-w=4.0:screen-w=1.5:" +
      "material=glass:edge-material=chrome:edge-radius=0.02"
    )
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.objectType shouldBe "tesseract-sponge"
    spec.level shouldBe Some(1.0f)
    spec.x shouldBe 1.0f
    spec.y shouldBe 2.0f
    spec.z shouldBe 3.0f
    spec.size shouldBe 2.5f
    spec.color shouldBe defined
    spec.material shouldBe defined
    spec.edgeMaterial shouldBe defined
    spec.edgeRadius shouldBe Some(0.02f)

    val proj = spec.projection4D.get
    proj.rotXW shouldBe 30f
    proj.rotYW shouldBe 20f
    proj.rotZW shouldBe 10f
    proj.eyeW shouldBe 4.0f
    proj.screenW shouldBe 1.5f

  it should "parse tesseract-sponge-2 with all parameters" in:
    val result = ObjectSpec.parse(
      "type=tesseract-sponge-2:level=2:pos=1,2,3:size=2.0:color=#FF44FF:" +
      "rot-xw=45:rot-yw=30:eye-w=5.0:screen-w=2.0"
    )
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.objectType shouldBe "tesseract-sponge-2"
    spec.level shouldBe Some(2.0f)
    spec.size shouldBe 2.0f

  // === 4D Validation Tests ===

  "4D validation" should "reject invalid eye-w and screen-w combination" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1:eye-w=1.0:screen-w=2.0")
    result shouldBe a[Left[?, ?]]
    result.left.getOrElse("") should include("must be greater than")

  it should "reject equal eye-w and screen-w" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1:eye-w=2.0:screen-w=2.0")
    result shouldBe a[Left[?, ?]]

  it should "reject negative projection values" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1:eye-w=-1:screen-w=-2")
    result shouldBe a[Left[?, ?]]
    result.left.getOrElse("") should include("must be positive")

  // === Sprint 9 Fix Tests ===

  "Sprint 9 fixes" should "support edge rendering parameters for tesseract-sponge" in:
    // Issue 2: Dynamic instance calculation for edge rendering
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1:edge-material=chrome:edge-radius=0.015")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.hasEdgeRendering shouldBe true
    spec.edgeMaterial shouldBe defined
    spec.edgeRadius shouldBe Some(0.015f)

  it should "support edge rendering parameters for tesseract-sponge-2" in:
    val result = ObjectSpec.parse("type=tesseract-sponge-2:level=1:edge-material=chrome:edge-radius=0.02")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.hasEdgeRendering shouldBe true
    spec.edgeMaterial shouldBe defined
    spec.edgeRadius shouldBe Some(0.02f)

  it should "parse multiple different 4D sponge types together" in:
    // Issue 3: Mix different 4D types
    val specs = ObjectSpec.parseAll(List(
      "type=tesseract-sponge:level=1:pos=-1,0,0",
      "type=tesseract-sponge-2:level=1:pos=1,0,0"
    ))
    specs shouldBe a[Right[?, ?]]

    val specList = specs.toOption.get
    specList should have length 2
    specList(0).objectType shouldBe "tesseract-sponge"
    specList(1).objectType shouldBe "tesseract-sponge-2"

  it should "parse tesseract with tesseract-sponge types together" in:
    val specs = ObjectSpec.parseAll(List(
      "type=tesseract:pos=-2,0,0",
      "type=tesseract-sponge:level=1:pos=0,0,0",
      "type=tesseract-sponge-2:level=1:pos=2,0,0"
    ))
    specs shouldBe a[Right[?, ?]]

    val specList = specs.toOption.get
    specList should have length 3
    ObjectType.isProjected4D(specList(0).objectType) shouldBe true
    ObjectType.isProjected4D(specList(1).objectType) shouldBe true
    ObjectType.isProjected4D(specList(2).objectType) shouldBe true

  it should "parse mixed 4D sponge and sphere objects" in:
    // Issue 4: Mix 4D + 3D objects
    val specs = ObjectSpec.parseAll(List(
      "type=tesseract-sponge-2:level=1:pos=-1.5,0,0:material=glass",
      "type=sphere:pos=1.5,0,0:material=chrome"
    ))
    specs shouldBe a[Right[?, ?]]

    val specList = specs.toOption.get
    specList should have length 2
    ObjectType.isProjected4D(specList(0).objectType) shouldBe true
    specList(1).objectType shouldBe "sphere"

  it should "support 4D rotation parameters for all 4D types" in:
    // Issue 1: 4D rotation detection works for all types
    val tesseractResult = ObjectSpec.parse("type=tesseract:rot-xw=45:rot-yw=30")
    tesseractResult shouldBe a[Right[?, ?]]
    tesseractResult.toOption.get.projection4D.get.rotXW shouldBe 45f

    val spongeResult = ObjectSpec.parse("type=tesseract-sponge:level=1:rot-xw=45:rot-yw=30")
    spongeResult shouldBe a[Right[?, ?]]
    spongeResult.toOption.get.projection4D.get.rotXW shouldBe 45f

    val sponge2Result = ObjectSpec.parse("type=tesseract-sponge-2:level=1:rot-xw=45:rot-yw=30")
    sponge2Result shouldBe a[Right[?, ?]]
    sponge2Result.toOption.get.projection4D.get.rotXW shouldBe 45f
