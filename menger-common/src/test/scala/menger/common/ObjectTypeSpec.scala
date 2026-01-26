package menger.common

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ObjectTypeSpec extends AnyFlatSpec with Matchers:

  // === Basic Type Validation Tests ===

  "ObjectType.isValid" should "accept all standard types" in:
    ObjectType.isValid("sphere") shouldBe true
    ObjectType.isValid("cube") shouldBe true
    ObjectType.isValid("sponge-volume") shouldBe true
    ObjectType.isValid("sponge-surface") shouldBe true
    ObjectType.isValid("cube-sponge") shouldBe true
    ObjectType.isValid("tesseract") shouldBe true
    ObjectType.isValid("tesseract-sponge") shouldBe true
    ObjectType.isValid("tesseract-sponge-2") shouldBe true

  it should "be case-insensitive" in:
    ObjectType.isValid("SPHERE") shouldBe true
    ObjectType.isValid("Tesseract") shouldBe true
    ObjectType.isValid("TESSERACT-SPONGE") shouldBe true
    ObjectType.isValid("Tesseract-Sponge-2") shouldBe true

  it should "reject invalid types" in:
    ObjectType.isValid("unknown") shouldBe false
    ObjectType.isValid("box") shouldBe false
    ObjectType.isValid("") shouldBe false
    ObjectType.isValid("tesseract-cube") shouldBe false

  // === Sponge Classification Tests ===

  "ObjectType.isSponge" should "identify 3D sponge types" in:
    ObjectType.isSponge("sponge-volume") shouldBe true
    ObjectType.isSponge("sponge-surface") shouldBe true
    ObjectType.isSponge("cube-sponge") shouldBe true

  it should "be case-insensitive" in:
    ObjectType.isSponge("SPONGE-VOLUME") shouldBe true
    ObjectType.isSponge("Sponge-Surface") shouldBe true

  it should "not classify non-sponge types as sponges" in:
    ObjectType.isSponge("sphere") shouldBe false
    ObjectType.isSponge("cube") shouldBe false
    ObjectType.isSponge("tesseract") shouldBe false
    ObjectType.isSponge("tesseract-sponge") shouldBe false
    ObjectType.isSponge("tesseract-sponge-2") shouldBe false

  // === Hypercube Classification Tests ===

  "ObjectType.isHypercube" should "identify 4D hypercube types" in:
    ObjectType.isHypercube("tesseract") shouldBe true
    ObjectType.isHypercube("tesseract-sponge") shouldBe true
    ObjectType.isHypercube("tesseract-sponge-2") shouldBe true

  it should "be case-insensitive" in:
    ObjectType.isHypercube("TESSERACT") shouldBe true
    ObjectType.isHypercube("Tesseract-Sponge") shouldBe true
    ObjectType.isHypercube("TESSERACT-SPONGE-2") shouldBe true

  it should "not classify 3D types as hypercubes" in:
    ObjectType.isHypercube("sphere") shouldBe false
    ObjectType.isHypercube("cube") shouldBe false
    ObjectType.isHypercube("sponge-volume") shouldBe false
    ObjectType.isHypercube("sponge-surface") shouldBe false
    ObjectType.isHypercube("cube-sponge") shouldBe false

  // === 4D Sponge Classification Tests ===

  "ObjectType.is4DSponge" should "identify 4D sponge types" in:
    ObjectType.is4DSponge("tesseract-sponge") shouldBe true
    ObjectType.is4DSponge("tesseract-sponge-2") shouldBe true

  it should "be case-insensitive" in:
    ObjectType.is4DSponge("TESSERACT-SPONGE") shouldBe true
    ObjectType.is4DSponge("Tesseract-Sponge-2") shouldBe true
    ObjectType.is4DSponge("TESSERACT-SPONGE-2") shouldBe true

  it should "not classify non-4D-sponge types" in:
    ObjectType.is4DSponge("tesseract") shouldBe false
    ObjectType.is4DSponge("sponge-volume") shouldBe false
    ObjectType.is4DSponge("sponge-surface") shouldBe false
    ObjectType.is4DSponge("cube-sponge") shouldBe false
    ObjectType.is4DSponge("sphere") shouldBe false
    ObjectType.is4DSponge("cube") shouldBe false

  // === Sponge or Cube Classification Tests ===

  "ObjectType.isSpongeOrCube" should "identify sponges and cubes" in:
    ObjectType.isSpongeOrCube("cube") shouldBe true
    ObjectType.isSpongeOrCube("sponge-volume") shouldBe true
    ObjectType.isSpongeOrCube("sponge-surface") shouldBe true
    ObjectType.isSpongeOrCube("cube-sponge") shouldBe true

  it should "not classify other types" in:
    ObjectType.isSpongeOrCube("sphere") shouldBe false
    ObjectType.isSpongeOrCube("tesseract") shouldBe false
    ObjectType.isSpongeOrCube("tesseract-sponge") shouldBe false
    ObjectType.isSpongeOrCube("tesseract-sponge-2") shouldBe false

  // === Valid Types String Tests ===

  "ObjectType.validTypesString" should "contain all valid types" in:
    val validStr = ObjectType.validTypesString
    validStr should include("sphere")
    validStr should include("cube")
    validStr should include("sponge-volume")
    validStr should include("sponge-surface")
    validStr should include("cube-sponge")
    validStr should include("tesseract")
    validStr should include("tesseract-sponge")
    validStr should include("tesseract-sponge-2")

  // === Mutual Exclusivity Tests ===

  "Type classification" should "have mutually exclusive sponge and hypercube categories" in:
    // 3D sponges are not hypercubes
    ObjectType.isSponge("sponge-volume") shouldBe true
    ObjectType.isHypercube("sponge-volume") shouldBe false

    // 4D sponges are hypercubes but not 3D sponges
    ObjectType.is4DSponge("tesseract-sponge") shouldBe true
    ObjectType.isHypercube("tesseract-sponge") shouldBe true
    ObjectType.isSponge("tesseract-sponge") shouldBe false

    // Regular tesseract is hypercube but not a sponge
    ObjectType.isHypercube("tesseract") shouldBe true
    ObjectType.isSponge("tesseract") shouldBe false
    ObjectType.is4DSponge("tesseract") shouldBe false

  it should "classify 4D sponges as both is4DSponge and isHypercube" in:
    ObjectType.is4DSponge("tesseract-sponge") shouldBe true
    ObjectType.isHypercube("tesseract-sponge") shouldBe true

    ObjectType.is4DSponge("tesseract-sponge-2") shouldBe true
    ObjectType.isHypercube("tesseract-sponge-2") shouldBe true

  // === Constants Access Tests ===

  "ObjectType.VALID_TYPES" should "contain all expected types" in:
    ObjectType.VALID_TYPES should contain("sphere")
    ObjectType.VALID_TYPES should contain("cube")
    ObjectType.VALID_TYPES should contain("sponge-volume")
    ObjectType.VALID_TYPES should contain("sponge-surface")
    ObjectType.VALID_TYPES should contain("cube-sponge")
    ObjectType.VALID_TYPES should contain("tesseract")
    ObjectType.VALID_TYPES should contain("tesseract-sponge")
    ObjectType.VALID_TYPES should contain("tesseract-sponge-2")
    ObjectType.VALID_TYPES.size shouldBe 8

  "ObjectType.SPONGE_TYPES" should "contain only 3D sponge types" in:
    ObjectType.SPONGE_TYPES should contain("sponge-volume")
    ObjectType.SPONGE_TYPES should contain("sponge-surface")
    ObjectType.SPONGE_TYPES should contain("cube-sponge")
    ObjectType.SPONGE_TYPES.size shouldBe 3

  "ObjectType.HYPERCUBE_TYPES" should "contain all 4D types" in:
    ObjectType.HYPERCUBE_TYPES should contain("tesseract")
    ObjectType.HYPERCUBE_TYPES should contain("tesseract-sponge")
    ObjectType.HYPERCUBE_TYPES should contain("tesseract-sponge-2")
    ObjectType.HYPERCUBE_TYPES.size shouldBe 3
