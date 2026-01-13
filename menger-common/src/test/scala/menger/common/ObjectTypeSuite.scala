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

  "ObjectType.isHypercube" should "classify tesseract as hypercube" in:
    ObjectType.isHypercube("tesseract") shouldBe true

  it should "classify tesseract case-insensitively" in:
    ObjectType.isHypercube("TESSERACT") shouldBe true
    ObjectType.isHypercube("Tesseract") shouldBe true

  it should "not classify cube as hypercube" in:
    ObjectType.isHypercube("cube") shouldBe false

  it should "not classify sphere as hypercube" in:
    ObjectType.isHypercube("sphere") shouldBe false

  it should "not classify sponge types as hypercube" in:
    ObjectType.isHypercube("sponge-volume") shouldBe false
    ObjectType.isHypercube("sponge-surface") shouldBe false
    ObjectType.isHypercube("cube-sponge") shouldBe false

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
