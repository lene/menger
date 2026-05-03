package menger.common

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ObjectTypeSuite extends AnyFlatSpec with Matchers:

  "ObjectType.isValid" should "recognize tesseract as valid" in:
    ObjectType.isValid("tesseract") shouldBe true

  it should "recognize tesseract case-insensitively" in:
    ObjectType.isValid("TESSERACT") shouldBe true
    ObjectType.isValid("Tesseract") shouldBe true
    ObjectType.isValid("TeSsErAcT") shouldBe true

  it should "recognize all standard types" in:
    ObjectType.isValid("sphere") shouldBe true
    ObjectType.isValid("cube") shouldBe true
    ObjectType.isValid("sponge-volume") shouldBe true
    ObjectType.isValid("sponge-surface") shouldBe true
    ObjectType.isValid("cube-sponge") shouldBe true

  it should "reject invalid types" in:
    ObjectType.isValid("invalid") shouldBe false
    ObjectType.isValid("") shouldBe false
    ObjectType.isValid("hypercube") shouldBe false

  "ObjectType.isProjected4D" should "classify tesseract as hypercube" in:
    ObjectType.isProjected4D("tesseract") shouldBe true

  it should "classify tesseract case-insensitively" in:
    ObjectType.isProjected4D("TESSERACT") shouldBe true
    ObjectType.isProjected4D("Tesseract") shouldBe true

  it should "not classify cube as hypercube" in:
    ObjectType.isProjected4D("cube") shouldBe false

  it should "not classify sphere as hypercube" in:
    ObjectType.isProjected4D("sphere") shouldBe false

  it should "not classify sponge types as hypercube" in:
    ObjectType.isProjected4D("sponge-volume") shouldBe false
    ObjectType.isProjected4D("sponge-surface") shouldBe false
    ObjectType.isProjected4D("cube-sponge") shouldBe false

  "ObjectType.isSponge" should "not classify tesseract as sponge" in:
    ObjectType.isSponge("tesseract") shouldBe false

  "ObjectType.validTypesString" should "include tesseract" in:
    ObjectType.validTypesString should include("tesseract")

  it should "include all valid types" in:
    val typesString = ObjectType.validTypesString
    typesString should include("sphere")
    typesString should include("cube")
    typesString should include("tesseract")
    typesString should include("sponge-volume")

  "ObjectType.isTriangleMesh" should "return true for cube" in:
    ObjectType.isTriangleMesh("cube") shouldBe true

  it should "return true for parametric" in:
    ObjectType.isTriangleMesh("parametric") shouldBe true

  it should "return true for sponge-volume" in:
    ObjectType.isTriangleMesh("sponge-volume") shouldBe true

  it should "return true for sponge-surface" in:
    ObjectType.isTriangleMesh("sponge-surface") shouldBe true

  it should "return true for sponge-recursive-ias" in:
    ObjectType.isTriangleMesh("sponge-recursive-ias") shouldBe true

  it should "return true for tesseract" in:
    ObjectType.isTriangleMesh("tesseract") shouldBe true

  it should "return true for tesseract-sponge-volume" in:
    ObjectType.isTriangleMesh("tesseract-sponge-volume") shouldBe true

  it should "return false for sphere" in:
    ObjectType.isTriangleMesh("sphere") shouldBe false

  it should "return false for cube-sponge" in:
    ObjectType.isTriangleMesh("cube-sponge") shouldBe false

  "ObjectType.TRIANGLE_MESH_TYPES" should "contain cube" in:
    ObjectType.TRIANGLE_MESH_TYPES should contain ("cube")

  it should "contain all four new polytopes" in:
    ObjectType.TRIANGLE_MESH_TYPES should contain ("tetrahedron")
    ObjectType.TRIANGLE_MESH_TYPES should contain ("octahedron")
    ObjectType.TRIANGLE_MESH_TYPES should contain ("dodecahedron")
    ObjectType.TRIANGLE_MESH_TYPES should contain ("icosahedron")

  it should "not contain sphere" in:
    ObjectType.TRIANGLE_MESH_TYPES should not contain "sphere"

  it should "not contain cube-sponge" in:
    ObjectType.TRIANGLE_MESH_TYPES should not contain "cube-sponge"

  "ObjectType.isTriangleMesh" should "return true for tetrahedron" in:
    ObjectType.isTriangleMesh("tetrahedron") shouldBe true

  it should "return true for octahedron" in:
    ObjectType.isTriangleMesh("octahedron") shouldBe true

  it should "return true for dodecahedron" in:
    ObjectType.isTriangleMesh("dodecahedron") shouldBe true

  it should "return true for icosahedron" in:
    ObjectType.isTriangleMesh("icosahedron") shouldBe true

  "ObjectType.isValid" should "return true for tetrahedron" in:
    ObjectType.isValid("tetrahedron") shouldBe true

  it should "return true for octahedron" in:
    ObjectType.isValid("octahedron") shouldBe true

  it should "return true for dodecahedron" in:
    ObjectType.isValid("dodecahedron") shouldBe true

  it should "return true for icosahedron" in:
    ObjectType.isValid("icosahedron") shouldBe true
