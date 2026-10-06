package menger.engines.scene

import menger.ObjectSpec
import menger.common.Material
import menger.common.ProfilingConfig
import menger.common.ObjectType
import menger.common.Vector
import menger.objects.Cube
import menger.objects.HoleCaps
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

  // menger#52: each 4D spec is projected with its own parameters.
  it should "allow 4D types with different projection parameters" in:
    val spec1 = ObjectSpec.parse("type=tesseract:rot-xw=45").toOption.get
    val spec2 = ObjectSpec.parse("type=tesseract-sponge:level=1:rot-xw=30").toOption.get
    builder.isCompatible(spec1, spec2) shouldBe true

  it should "allow mixed 4D and non-4D types (TD-5: each spec gets its own mesh+GAS)" in:
    val spec1 = ObjectSpec.parse("type=tesseract").toOption.get
    val spec2 = ObjectSpec.parse("type=cube").toOption.get
    builder.isCompatible(spec1, spec2) shouldBe true

  it should "allow cube + sponge (TD-5: distinct triangle-mesh types coexist)" in:
    val spec1 = ObjectSpec.parse("type=cube").toOption.get
    val spec2 = ObjectSpec.parse("type=sponge-volume:level=1").toOption.get
    builder.isCompatible(spec1, spec2) shouldBe true

  it should "allow multiple cubes" in:
    val spec1 = ObjectSpec.parse("type=cube").toOption.get
    val spec2 = ObjectSpec.parse("type=cube").toOption.get
    builder.isCompatible(spec1, spec2) shouldBe true

  it should "allow sponges of same level" in:
    val spec1 = ObjectSpec.parse("type=sponge-volume:level=2").toOption.get
    val spec2 = ObjectSpec.parse("type=sponge-volume:level=2").toOption.get
    builder.isCompatible(spec1, spec2) shouldBe true

  it should "allow sponges of different levels" in:
    val spec1 = ObjectSpec.parse("type=sponge-volume:level=1").toOption.get
    val spec2 = ObjectSpec.parse("type=sponge-volume:level=2").toOption.get
    builder.isCompatible(spec1, spec2) shouldBe true

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

  it should "accept cube + tesseract (TD-5 resolved: distinct mesh types coexist)" in:
    val specs = List(
      ObjectSpec.parse("type=cube").toOption.get,
      ObjectSpec.parse("type=tesseract").toOption.get
    )
    builder.validate(specs, 100) shouldBe Right(())

  it should "accept 4D specs with different projection parameters (menger#52)" in:
    val specs = List(
      ObjectSpec.parse("type=tesseract:rot-xw=45").toOption.get,
      ObjectSpec.parse("type=tesseract-sponge:level=1:rot-xw=30").toOption.get
    )
    builder.validate(specs, 100) shouldBe Right(())

  it should "accept fractional level for sponge-recursive-ias" in:
    val spec = ObjectSpec.parse("type=sponge-recursive-ias:level=2.5").toOption.get
    builder.validate(List(spec), 100) shouldBe Right(())

  it should "reject level >= 14 for sponge-recursive-ias" in:
    val spec = ObjectSpec.parse("type=sponge-recursive-ias:level=14").toOption.get
    val result = builder.validate(List(spec), 100)
    result shouldBe a[Left[?, ?]]

  // === calculateInstanceCount Tests ===

  "TriangleMeshSceneBuilder.calculateInstanceCount" should "return 2 for fractional sponge-recursive-ias" in:
    val spec = ObjectSpec.parse("type=sponge-recursive-ias:level=2.5").toOption.get
    builder.calculateInstanceCount(List(spec)) shouldBe 2L

  it should "return 1 for integer sponge-recursive-ias" in:
    val spec = ObjectSpec.parse("type=sponge-recursive-ias:level=2").toOption.get
    builder.calculateInstanceCount(List(spec)) shouldBe 1L

  // === validate: instance count ===

  "TriangleMeshSceneBuilder.validate" should "reject too many instances" in:
    val specs = List.fill(100)(ObjectSpec.parse("type=cube").toOption.get)
    val result = builder.validate(specs, 50)
    result shouldBe a[Left[?, ?]]
    result.left.getOrElse("") should include("Too many")

  // === recursive-IAS fractional levels (menger#55) ===

  // Usability review 2026-09, session 2 (F35 leftover): a fractional recursive-IAS sponge laid
  // the whole coarse level over the fine one at alpha 1 - frac; only its hole caps should fade.
  private val cube = Cube(center = Vector.Zero[3], scale = 1f).toTriangleMesh

  "TriangleMeshSceneBuilder.recursiveIASInstances" should
    "add the fine level on the cube and fade only the coarse level's hole caps" in:
      val instances = TriangleMeshSceneBuilder.recursiveIASInstances(1.25f, cube, Material.Film)
      instances should have size 2
      val (fineLeaf, fineLevel, fineMaterial, fineCoverage) = instances.head
      fineLeaf shouldBe cube
      fineLevel shouldBe 2
      fineMaterial shouldBe Material.Film
      fineCoverage shouldBe 1f
      val (coarseLeaf, coarseLevel, coarseMaterial, coarseCoverage) = instances(1)
      coarseLeaf.vertices.toSeq shouldBe HoleCaps.of(cube).vertices.toSeq
      coarseLevel shouldBe 1
      // menger#56: the caps fade by instance coverage and keep their material, so a refractive
      // material keeps its absorption (scaling alpha could not fade glass or film caps).
      coarseMaterial shouldBe Material.Film
      coarseCoverage shouldBe (0.75f +- 1e-6f)

  it should "add only the plain cube for an integer level" in:
    TriangleMeshSceneBuilder.recursiveIASInstances(2f, cube, Material.Film) shouldBe
      List((cube, 2, Material.Film, 1f))

  "TriangleMeshSceneBuilder.holeCapsCoverage" should "be 1 minus the fractional part" in:
    TriangleMeshSceneBuilder.holeCapsCoverage(1.25f) shouldBe (0.75f +- 1e-6f)
    TriangleMeshSceneBuilder.holeCapsCoverage(2.9f) shouldBe (0.1f +- 1e-5f)
