package menger.engines

import menger.ObjectSpec
import menger.common.ObjectType
import menger.common.ProfilingConfig
import menger.engines.scene.CubeSpongeSceneBuilder
import menger.engines.scene.IFS4DType
import menger.engines.scene.Instanced4DSceneBuilder
import menger.engines.scene.SphereSceneBuilder
import menger.engines.scene.TesseractEdgeSceneBuilder
import menger.engines.scene.TriangleMeshSceneBuilder
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GeometryRegistrySuite extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig.disabled

  private def spec(t: String) = ObjectSpec(objectType = t, x = 0, y = 0, z = 0, size = 1)

  "GeometryRegistry.builderFor" should "return SphereSceneBuilder for all-sphere specs" in:
    val result = GeometryRegistry.builderFor(List(spec("sphere")))
    result shouldBe defined
    result.get shouldBe a [SphereSceneBuilder]

  it should "return SphereSceneBuilder for multiple sphere specs" in:
    val result = GeometryRegistry.builderFor(List(spec("sphere"), spec("sphere")))
    result shouldBe defined
    result.get shouldBe a [SphereSceneBuilder]

  it should "return CubeSpongeSceneBuilder for all-cube-sponge specs" in:
    val result = GeometryRegistry.builderFor(List(spec("cube-sponge")))
    result shouldBe defined
    result.get shouldBe a [CubeSpongeSceneBuilder]

  it should "return TriangleMeshSceneBuilder for cube specs" in:
    val result = GeometryRegistry.builderFor(List(spec("cube")))
    result shouldBe defined
    result.get shouldBe a [TriangleMeshSceneBuilder]

  it should "return TriangleMeshSceneBuilder for sponge-volume specs" in:
    val result = GeometryRegistry.builderFor(List(spec("sponge-volume")))
    result shouldBe defined
    result.get shouldBe a [TriangleMeshSceneBuilder]

  it should "return TriangleMeshSceneBuilder for sponge-recursive-ias specs" in:
    val result = GeometryRegistry.builderFor(List(spec("sponge-recursive-ias")))
    result shouldBe defined
    result.get shouldBe a [TriangleMeshSceneBuilder]

  it should "return TriangleMeshSceneBuilder for tesseract specs without edge rendering" in:
    val result = GeometryRegistry.builderFor(List(spec("tesseract")))
    result shouldBe defined
    result.get shouldBe a [TriangleMeshSceneBuilder]

  it should "return TesseractEdgeSceneBuilder for tesseract specs with edge rendering" in:
    val edgeSpec = ObjectSpec(
      objectType = "tesseract", x = 0, y = 0, z = 0, size = 1, edgeRadius = Some(0.02f)
    )
    val result = GeometryRegistry.builderFor(List(edgeSpec))
    result shouldBe defined
    result.get shouldBe a [TesseractEdgeSceneBuilder]

  it should "return Instanced4DSceneBuilder for menger4d specs" in:
    val result = GeometryRegistry.builderFor(List(spec("menger4d")))
    result shouldBe defined
    result.get match
      case b: Instanced4DSceneBuilder => b.ifsType shouldBe IFS4DType.Menger4D
      case other => fail(s"expected Instanced4DSceneBuilder, got ${other.getClass.getSimpleName}")

  it should "return Instanced4DSceneBuilder for sierpinski4d specs" in:
    val result = GeometryRegistry.builderFor(List(spec("sierpinski4d")))
    result shouldBe defined
    result.get match
      case b: Instanced4DSceneBuilder => b.ifsType shouldBe IFS4DType.Sierpinski4D
      case other => fail(s"expected Instanced4DSceneBuilder, got ${other.getClass.getSimpleName}")

  it should "return Instanced4DSceneBuilder for hexadecachoron4d specs" in:
    val result = GeometryRegistry.builderFor(List(spec("hexadecachoron4d")))
    result shouldBe defined
    result.get match
      case b: Instanced4DSceneBuilder => b.ifsType shouldBe IFS4DType.Hexadecachoron4D
      case other => fail(s"expected Instanced4DSceneBuilder, got ${other.getClass.getSimpleName}")

  // Fitness function: every instanced-4D type must resolve to a dedicated builder.
  // Guards against the dispatch drift where sierpinski4d/hexadecachoron4d were wired
  // in InteractiveEngine but fell through to None here (non-interactive render path).
  it should "resolve every instanced-4D valid type to a builder" in:
    val instanced4D = ObjectType.VALID_TYPES.filter(t =>
      ObjectType.isMenger4D(t) || ObjectType.isSierpinski4D(t) || ObjectType.isHexadecachoron4D(t))
    instanced4D should not be empty
    instanced4D.foreach: t =>
      withClue(s"no builder for instanced-4D type '$t': "):
        GeometryRegistry.builderFor(List(spec(t))) shouldBe defined

  it should "return None for mixed sphere + cube specs" in:
    val result = GeometryRegistry.builderFor(List(spec("sphere"), spec("cube")))
    result shouldBe None

  it should "return None for empty spec list" in:
    val result = GeometryRegistry.builderFor(List.empty)
    result shouldBe None

  // Fitness function (T1): every VALID_TYPES entry handled by TypeRegistry
  // must resolve to exactly one builder. Guards against dispatch drift.
  // Note: lsystem is in VALID_TYPES but lacks a dedicated SceneBuilder
  // (tracked as Sprint 31 deferred item).
  it should "resolve every registered type to exactly one builder" in:
    val registeredTypes = TypeRegistry.builtInTypeNames ++
      ObjectType.VALID_TYPES.filter(TypeRegistry.isTriangleMeshType)
    registeredTypes.foreach: t =>
      withClue(s"no builder for type '$t': "):
        val builder = GeometryRegistry.builderFor(List(spec(t)))
        builder shouldBe defined
        builder.get should not be null

  // Fitness function (F8, Sprint 35 Ph4): every VALID_TYPES entry resolves via
  // TypeRegistry.forType directly (not just indirectly through builderFor above).
  // lsystem is the one exception: it is a builtInTypeNames entry but is not itself
  // a VALID_TYPES member requiring forType resolution via the triangle-mesh path —
  // included here since it IS in VALID_TYPES and IS a builtin entry.
  it should "resolve every ObjectType.VALID_TYPES entry via TypeRegistry.forType" in:
    ObjectType.VALID_TYPES.foreach: t =>
      withClue(s"TypeRegistry.forType has no entry for '$t': "):
        TypeRegistry.forType(t) shouldBe defined

  // Regression guard for the InteractiveEngine 4D fast-path unification (F8): the new
  // single predicate must agree with the old hand-rolled 4-way OR for every valid type.
  it should "agree with the legacy 4-way OR for is4DFastPathType, for every valid type" in:
    def legacy4DCheck(t: String): Boolean =
      ObjectType.isProjected4D(t) || ObjectType.isMenger4D(t) ||
      ObjectType.isSierpinski4D(t) || ObjectType.isHexadecachoron4D(t)
    ObjectType.VALID_TYPES.foreach: t =>
      withClue(s"is4DFastPathType('$t') disagrees with the legacy check: "):
        TypeRegistry.is4DFastPathType(t) shouldBe legacy4DCheck(t)
