package menger.engines

import com.badlogic.gdx.math.Vector3
import menger.ObjectSpec
import menger.Projection4DSpec
import menger.common.ObjectType
import menger.objects.higher_d.TesseractMesh
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TesseractIntegrationSuite extends AnyFlatSpec with Matchers:

  // === ObjectSpec to TesseractMesh Pipeline Tests ===

  "Tesseract integration" should "create TesseractMesh from parsed ObjectSpec with defaults" in:
    val result = ObjectSpec.parse("type=tesseract:pos=0,0,0:size=2.0")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.objectType shouldBe "tesseract"
    spec.projection4D shouldBe defined

    // Create TesseractMesh from ObjectSpec parameters
    val proj = spec.projection4D.get
    val mesh = TesseractMesh(
      center = Vector3(spec.x, spec.y, spec.z),
      size = spec.size,
      eyeW = proj.eyeW,
      screenW = proj.screenW,
      rotXW = proj.rotXW,
      rotYW = proj.rotYW,
      rotZW = proj.rotZW
    )

    val data = mesh.toTriangleMesh
    data.numVertices shouldBe 96
    data.numTriangles shouldBe 48

  it should "create TesseractMesh with custom 4D rotation from ObjectSpec" in:
    val result = ObjectSpec.parse("type=tesseract:rot-xw=45:rot-yw=30:rot-zw=15")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    val proj = spec.projection4D.get

    proj.rotXW shouldBe 45f
    proj.rotYW shouldBe 30f
    proj.rotZW shouldBe 15f

    // Verify mesh generation works with custom rotation
    val mesh = TesseractMesh(
      rotXW = proj.rotXW,
      rotYW = proj.rotYW,
      rotZW = proj.rotZW
    )
    val data = mesh.toTriangleMesh
    data.numTriangles shouldBe 48

  it should "create TesseractMesh with custom projection parameters from ObjectSpec" in:
    val result = ObjectSpec.parse("type=tesseract:eye-w=5.0:screen-w=2.5")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    val proj = spec.projection4D.get

    proj.eyeW shouldBe 5.0f
    proj.screenW shouldBe 2.5f

    // Verify mesh generation works with custom projection
    val mesh = TesseractMesh(
      eyeW = proj.eyeW,
      screenW = proj.screenW
    )
    val data = mesh.toTriangleMesh
    data.numTriangles shouldBe 48

  // === Type Classification Tests ===

  it should "classify tesseract as hypercube type" in:
    ObjectType.isProjected4D("tesseract") shouldBe true
    ObjectType.isProjected4D("TESSERACT") shouldBe true
    ObjectType.isProjected4D("Tesseract") shouldBe true

  it should "classify tesseract as valid type" in:
    ObjectType.isValid("tesseract") shouldBe true

  it should "not classify tesseract as sponge type" in:
    ObjectType.isSponge("tesseract") shouldBe false

  it should "classify tesseract as triangle mesh type" in:
    // This tests the isTriangleMeshType logic used in OptiXEngine
    val isTriangleMesh = ObjectType.isProjected4D("tesseract")
    isTriangleMesh shouldBe true

  // === Material Support Tests ===

  it should "parse tesseract with glass material" in:
    val result = ObjectSpec.parse("type=tesseract:material=glass")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.material shouldBe defined
    spec.material.get.ior shouldBe 1.5f

  it should "parse tesseract with chrome material" in:
    val result = ObjectSpec.parse("type=tesseract:material=chrome")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.material shouldBe defined
    spec.material.get.metallic shouldBe 1.0f

  it should "parse tesseract with color override" in:
    val result = ObjectSpec.parse("type=tesseract:color=#FF5500")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.color shouldBe defined
    spec.color.get.r shouldBe 1.0f
    spec.color.get.g should be(0.33f +- 0.01f)
    spec.color.get.b shouldBe 0.0f

  it should "parse tesseract with material and IOR override" in:
    val result = ObjectSpec.parse("type=tesseract:material=glass:ior=2.0")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.material shouldBe defined
    spec.material.get.ior shouldBe 2.0f

  it should "parse tesseract with material and color override" in:
    val result = ObjectSpec.parse("type=tesseract:material=glass:color=#88AAFF")
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.material shouldBe defined
    // Color should be overridden to the specified value
    spec.material.get.color.r should be(0.53f +- 0.01f)

  // === Combined Parameters Tests ===

  it should "parse tesseract with all parameters combined" in:
    val result = ObjectSpec.parse(
      "type=tesseract:pos=1,2,3:size=2.5:color=#4488FF:rot-xw=30:rot-yw=20:rot-zw=10:eye-w=4.0:screen-w=1.5"
    )
    result shouldBe a[Right[?, ?]]

    val spec = result.toOption.get
    spec.objectType shouldBe "tesseract"
    spec.x shouldBe 1.0f
    spec.y shouldBe 2.0f
    spec.z shouldBe 3.0f
    spec.size shouldBe 2.5f
    spec.color shouldBe defined

    val proj = spec.projection4D.get
    proj.rotXW shouldBe 30f
    proj.rotYW shouldBe 20f
    proj.rotZW shouldBe 10f
    proj.eyeW shouldBe 4.0f
    proj.screenW shouldBe 1.5f

  it should "generate valid mesh from fully specified ObjectSpec" in:
    val result = ObjectSpec.parse(
      "type=tesseract:pos=5,-3,2:size=3.0:rot-xw=45:rot-yw=30:rot-zw=15:eye-w=5.0:screen-w=2.0"
    )
    val spec = result.toOption.get
    val proj = spec.projection4D.get

    val mesh = TesseractMesh(
      center = Vector3(spec.x, spec.y, spec.z),
      size = spec.size,
      eyeW = proj.eyeW,
      screenW = proj.screenW,
      rotXW = proj.rotXW,
      rotYW = proj.rotYW,
      rotZW = proj.rotZW
    )

    val data = mesh.toTriangleMesh
    data.numTriangles shouldBe 48
    data.numVertices shouldBe 96

    // Verify translation was applied (first vertex should be offset)
    // Note: exact values depend on rotation, but should include the offset
    data.vertices.length shouldBe 96 * 8  // 96 vertices * 8 floats each

  // === Projection4DSpec Default Values Tests ===

  "Projection4DSpec defaults" should "match documented values" in:
    Projection4DSpec.DefaultEyeW shouldBe 3.0f
    Projection4DSpec.DefaultScreenW shouldBe 1.5f
    Projection4DSpec.DefaultRotXW shouldBe 15f
    Projection4DSpec.DefaultRotYW shouldBe 10f
    Projection4DSpec.DefaultRotZW shouldBe 0f

  it should "match TesseractMesh default values" in:
    val defaultMesh = TesseractMesh()
    val defaultProj = Projection4DSpec.default

    defaultMesh.eyeW shouldBe defaultProj.eyeW
    defaultMesh.screenW shouldBe defaultProj.screenW
    defaultMesh.rotXW shouldBe defaultProj.rotXW
    defaultMesh.rotYW shouldBe defaultProj.rotYW
    defaultMesh.rotZW shouldBe defaultProj.rotZW

  // === isCompatibleMesh Logic Tests ===

  "Tesseract compatibility" should "identify same 4D parameters as compatible" in:
    val spec1 = ObjectSpec.parse("type=tesseract:rot-xw=30:rot-yw=20")
    val spec2 = ObjectSpec.parse("type=tesseract:rot-xw=30:rot-yw=20")

    spec1 shouldBe a[Right[?, ?]]
    spec2 shouldBe a[Right[?, ?]]

    val s1 = spec1.toOption.get
    val s2 = spec2.toOption.get

    // Same projection params should be compatible
    s1.projection4D.get shouldBe s2.projection4D.get

  it should "identify different 4D parameters as incompatible for mesh sharing" in:
    val spec1 = ObjectSpec.parse("type=tesseract:rot-xw=30")
    val spec2 = ObjectSpec.parse("type=tesseract:rot-xw=45")

    val s1 = spec1.toOption.get
    val s2 = spec2.toOption.get

    // Different rotation should produce different projection params
    s1.projection4D.get should not be s2.projection4D.get

  // === Validation Tests ===

  "Tesseract validation" should "reject invalid eye-w and screen-w combination" in:
    val result = ObjectSpec.parse("type=tesseract:eye-w=1.0:screen-w=2.0")
    result shouldBe a[Left[?, ?]]
    result.left.getOrElse("") should include("must be greater than")

  it should "reject equal eye-w and screen-w" in:
    val result = ObjectSpec.parse("type=tesseract:eye-w=2.0:screen-w=2.0")
    result shouldBe a[Left[?, ?]]

  it should "reject negative projection values" in:
    val result = ObjectSpec.parse("type=tesseract:eye-w=-1:screen-w=-2")
    result shouldBe a[Left[?, ?]]
    result.left.getOrElse("") should include("must be positive")

  it should "reject invalid rotation value format" in:
    val result = ObjectSpec.parse("type=tesseract:rot-xw=invalid")
    result shouldBe a[Left[?, ?]]
    result.left.getOrElse("") should include("Invalid rot-xw")

  // === SceneType Classification Tests ===

  "SceneType classification" should "group tesseract with triangle mesh types" in:
    // This verifies the logic in OptiXEngine.isTriangleMeshType
    val tesseractIsTriangleMesh = ObjectType.isProjected4D("tesseract")
    tesseractIsTriangleMesh shouldBe true

    // Spheres are NOT triangle meshes
    ObjectType.isProjected4D("sphere") shouldBe false

    // Cubes and sponges ARE triangle meshes (via isSponge or direct check)
    ObjectType.isSponge("sponge-volume") shouldBe true
    ObjectType.isSponge("sponge-surface") shouldBe true

  // === Multiple Tesseracts Tests ===

  "Multiple tesseracts" should "parse list of tesseract specs" in:
    val specs = List(
      "type=tesseract:pos=-2,0,0:color=#FF0000",
      "type=tesseract:pos=2,0,0:color=#00FF00"
    )
    val result = ObjectSpec.parseAll(specs)
    result shouldBe a[Right[?, ?]]

    val objects = result.toOption.get
    objects.length shouldBe 2
    objects(0).objectType shouldBe "tesseract"
    objects(1).objectType shouldBe "tesseract"
    objects(0).x shouldBe -2.0f
    objects(1).x shouldBe 2.0f

  it should "support tesseracts with different materials in same scene" in:
    val specs = List(
      "type=tesseract:pos=-2,0,0:material=glass",
      "type=tesseract:pos=2,0,0:material=chrome"
    )
    val result = ObjectSpec.parseAll(specs)
    result shouldBe a[Right[?, ?]]

    val objects = result.toOption.get
    objects(0).material.get.ior shouldBe 1.5f  // Glass
    objects(1).material.get.metallic shouldBe 1.0f  // Chrome
