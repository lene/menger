package menger.engines.scene

import menger.ObjectSpec
import menger.ProfilingConfig
import menger.common.ObjectType
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TriangleMeshSceneBuilderSpec extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig.disabled

  val builder = TriangleMeshSceneBuilder(".")

  // === Compatibility Tests for Sprint 9 Fixes ===

  "TriangleMeshSceneBuilder.isCompatible" should "allow same 4D types with matching projection" in:
    val spec1 = ObjectSpec.parse("type=tesseract:rot-xw=45").toOption.get
    val spec2 = ObjectSpec.parse("type=tesseract:rot-xw=45").toOption.get
    builder.isCompatible(spec1, spec2) shouldBe true

  it should "allow different 4D types with matching projection parameters" in:
    // Issue 3: Mix different 4D types
    val spec1 = ObjectSpec.parse("type=tesseract:rot-xw=45:rot-yw=30").toOption.get
    val spec2 = ObjectSpec.parse("type=tesseract-sponge:level=1:rot-xw=45:rot-yw=30").toOption.get
    builder.isCompatible(spec1, spec2) shouldBe true

  it should "allow tesseract-sponge and tesseract-sponge-2 with matching projection" in:
    val spec1 = ObjectSpec.parse("type=tesseract-sponge:level=1").toOption.get
    val spec2 = ObjectSpec.parse("type=tesseract-sponge-2:level=1").toOption.get
    builder.isCompatible(spec1, spec2) shouldBe true

  it should "reject different 4D types with mismatched projection parameters" in:
    val spec1 = ObjectSpec.parse("type=tesseract:rot-xw=45").toOption.get
    val spec2 = ObjectSpec.parse("type=tesseract-sponge:level=1:rot-xw=30").toOption.get
    builder.isCompatible(spec1, spec2) shouldBe false

  it should "reject mixed 4D and non-4D types" in:
    val spec1 = ObjectSpec.parse("type=tesseract").toOption.get
    val spec2 = ObjectSpec.parse("type=cube").toOption.get
    builder.isCompatible(spec1, spec2) shouldBe false

  it should "allow multiple cubes" in:
    val spec1 = ObjectSpec.parse("type=cube").toOption.get
    val spec2 = ObjectSpec.parse("type=cube").toOption.get
    builder.isCompatible(spec1, spec2) shouldBe true

  it should "allow sponges of same level" in:
    val spec1 = ObjectSpec.parse("type=sponge-volume:level=2").toOption.get
    val spec2 = ObjectSpec.parse("type=sponge-volume:level=2").toOption.get
    builder.isCompatible(spec1, spec2) shouldBe true

  it should "reject sponges of different levels" in:
    val spec1 = ObjectSpec.parse("type=sponge-volume:level=1").toOption.get
    val spec2 = ObjectSpec.parse("type=sponge-volume:level=2").toOption.get
    builder.isCompatible(spec1, spec2) shouldBe false

  // === ObjectType classification for 4D types ===

  "ObjectType.isProjected4D" should "identify all 4D projected types" in:
    // Issue 1: Rotation detection uses this
    ObjectType.isProjected4D("tesseract") shouldBe true
    ObjectType.isProjected4D("tesseract-sponge") shouldBe true
    ObjectType.isProjected4D("tesseract-sponge-2") shouldBe true

  it should "not classify non-4D types" in:
    ObjectType.isProjected4D("sphere") shouldBe false
    ObjectType.isProjected4D("cube") shouldBe false
    ObjectType.isProjected4D("sponge-volume") shouldBe false

  // === Validation Tests ===

  "TriangleMeshSceneBuilder.validate" should "accept compatible 4D types" in:
    val specs = List(
      ObjectSpec.parse("type=tesseract").toOption.get,
      ObjectSpec.parse("type=tesseract-sponge:level=1").toOption.get
    )
    builder.validate(specs, 100) shouldBe Right(())

  it should "reject incompatible types" in:
    val specs = List(
      ObjectSpec.parse("type=cube").toOption.get,
      ObjectSpec.parse("type=tesseract").toOption.get
    )
    val result = builder.validate(specs, 100)
    result shouldBe a[Left[?, ?]]
    result.left.getOrElse("") should include("Incompatible")

  it should "reject too many instances" in:
    val specs = List.fill(100)(ObjectSpec.parse("type=cube").toOption.get)
    val result = builder.validate(specs, 50)
    result shouldBe a[Left[?, ?]]
    result.left.getOrElse("") should include("Too many")
