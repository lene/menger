package menger.engines.scene

import menger.ObjectSpec
import menger.common.ProfilingConfig
import menger.common.ObjectType
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

@SuppressWarnings(Array("org.wartremover.warts.AsInstanceOf"))
class FractionalLevelSceneBuilderSuite extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig(minDurationMs = None)

  // === Fractional Level Detection Tests ===

  "TriangleMeshSceneBuilder" should "detect fractional 4D sponge levels" in:
    val builder = TriangleMeshSceneBuilder(".")
    val integerSpec = ObjectSpec.parse("type=tesseract-sponge:level=1").toOption.get
    val fractionalSpec = ObjectSpec.parse("type=tesseract-sponge:level=1.5").toOption.get

    // Use reflection to access private method for testing
    val isFractional4DSpongeMethod = builder.getClass.getDeclaredMethod("isFractional4DSponge", classOf[ObjectSpec])
    isFractional4DSpongeMethod.setAccessible(true)

    isFractional4DSpongeMethod.invoke(builder, integerSpec).asInstanceOf[Boolean] shouldBe false
    isFractional4DSpongeMethod.invoke(builder, fractionalSpec).asInstanceOf[Boolean] shouldBe true

  it should "detect fractional levels for tesseract-sponge-2" in:
    val builder = TriangleMeshSceneBuilder(".")
    val fractionalSpec = ObjectSpec.parse("type=tesseract-sponge-2:level=2.7").toOption.get

    val isFractional4DSpongeMethod = builder.getClass.getDeclaredMethod("isFractional4DSponge", classOf[ObjectSpec])
    isFractional4DSpongeMethod.setAccessible(true)

    isFractional4DSpongeMethod.invoke(builder, fractionalSpec).asInstanceOf[Boolean] shouldBe true

  it should "not detect fractional levels for non-4D sponges" in:
    val builder = TriangleMeshSceneBuilder(".")
    val cubeSpec = ObjectSpec.parse("type=cube:size=1").toOption.get
    val spongeSpec = ObjectSpec.parse("type=sponge-volume:level=1.5").toOption.get
    val tesseractSpec = ObjectSpec.parse("type=tesseract").toOption.get

    val isFractional4DSpongeMethod = builder.getClass.getDeclaredMethod("isFractional4DSponge", classOf[ObjectSpec])
    isFractional4DSpongeMethod.setAccessible(true)

    isFractional4DSpongeMethod.invoke(builder, cubeSpec).asInstanceOf[Boolean] shouldBe false
    isFractional4DSpongeMethod.invoke(builder, spongeSpec).asInstanceOf[Boolean] shouldBe false
    isFractional4DSpongeMethod.invoke(builder, tesseractSpec).asInstanceOf[Boolean] shouldBe false

  // === Instance Count Calculation Tests ===

  "Instance count calculation" should "count 1 for integer level sponges" in:
    val builder = TriangleMeshSceneBuilder(".")
    val specs = List(
      ObjectSpec.parse("type=tesseract-sponge:level=1").toOption.get
    )

    builder.calculateInstanceCount(specs) shouldBe 1L

  it should "count 2 for fractional level sponges (GPU split)" in:
    val builder = TriangleMeshSceneBuilder(".")
    val specs = List(
      ObjectSpec.parse("type=tesseract-sponge:level=1.5").toOption.get
    )

    // GPU path: a fractional 4D sponge emits two instances (level n + level n+1).
    builder.calculateInstanceCount(specs) shouldBe 2L

  it should "count correctly for mixed integer and fractional levels" in:
    val builder = TriangleMeshSceneBuilder(".")
    val specs = List(
      ObjectSpec.parse("type=tesseract-sponge:level=1").toOption.get,
      ObjectSpec.parse("type=tesseract-sponge:level=1.5").toOption.get,
      ObjectSpec.parse("type=tesseract-sponge:level=2").toOption.get,
      ObjectSpec.parse("type=tesseract-sponge:level=2.3").toOption.get
    )

    // GPU split: 1 + 2 + 1 + 2 = 6 (fractional levels emit two instances each).
    builder.calculateInstanceCount(specs) shouldBe 6L

  it should "count correctly for multiple fractional levels" in:
    val builder = TriangleMeshSceneBuilder(".")
    val specs = List(
      ObjectSpec.parse("type=tesseract-sponge:level=1.2").toOption.get,
      ObjectSpec.parse("type=tesseract-sponge:level=1.5").toOption.get,
      ObjectSpec.parse("type=tesseract-sponge:level=1.8").toOption.get
    )

    // GPU split: each fractional spec emits two instances → 2 + 2 + 2 = 6.
    builder.calculateInstanceCount(specs) shouldBe 6L

  it should "count correctly for non-sponge objects" in:
    val builder = TriangleMeshSceneBuilder(".")
    val specs = List(
      ObjectSpec.parse("type=cube").toOption.get,
      ObjectSpec.parse("type=tesseract").toOption.get
    )

    builder.calculateInstanceCount(specs) shouldBe 2L

  // === Compatibility Tests ===

  "Compatibility checks" should "allow same type with different integer levels" in:
    val builder = TriangleMeshSceneBuilder(".")
    val spec1 = ObjectSpec.parse("type=tesseract-sponge:level=1").toOption.get
    val spec2 = ObjectSpec.parse("type=tesseract-sponge:level=2").toOption.get

    builder.isCompatible(spec1, spec2) shouldBe true

  it should "allow same type with integer and fractional levels" in:
    val builder = TriangleMeshSceneBuilder(".")
    val spec1 = ObjectSpec.parse("type=tesseract-sponge:level=1").toOption.get
    val spec2 = ObjectSpec.parse("type=tesseract-sponge:level=1.5").toOption.get

    builder.isCompatible(spec1, spec2) shouldBe true

  it should "allow same type with different fractional levels" in:
    val builder = TriangleMeshSceneBuilder(".")
    val spec1 = ObjectSpec.parse("type=tesseract-sponge:level=1.3").toOption.get
    val spec2 = ObjectSpec.parse("type=tesseract-sponge:level=2.7").toOption.get

    builder.isCompatible(spec1, spec2) shouldBe true

  it should "allow different 4D sponge types" in:
    val builder = TriangleMeshSceneBuilder(".")
    val spec1 = ObjectSpec.parse("type=tesseract-sponge:level=1").toOption.get
    val spec2 = ObjectSpec.parse("type=tesseract-sponge-2:level=1").toOption.get

    builder.isCompatible(spec1, spec2) shouldBe true

  it should "allow 3D sponges with different levels" in:
    val builder = TriangleMeshSceneBuilder(".")
    val spec1 = ObjectSpec.parse("type=sponge-volume:level=1.5").toOption.get
    val spec2 = ObjectSpec.parse("type=sponge-volume:level=1").toOption.get

    builder.isCompatible(spec1, spec2) shouldBe true

  it should "allow mixing sponge-volume and sponge-surface" in:
    val builder = TriangleMeshSceneBuilder(".")
    val spec1 =
      ObjectSpec.parse("type=sponge-volume:level=1.5").toOption.get
    val spec2 =
      ObjectSpec.parse("type=sponge-surface:level=1.5").toOption.get

    builder.isCompatible(spec1, spec2) shouldBe true

  it should "reject 4D sponge without level parameter" in:
    val builder = TriangleMeshSceneBuilder(".")
    // These should fail at parsing, but let's test the compatibility logic
    // by constructing specs directly (which bypasses validation)
    val spec1 = ObjectSpec.parse("type=tesseract-sponge:level=1").toOption.get
    val spec2 = spec1.copy(level = None)

    builder.isCompatible(spec1, spec2) shouldBe false

  // === Edge Cases ===

  "Edge cases" should "handle level 0.0 as integer" in:
    val builder = TriangleMeshSceneBuilder(".")
    val spec = ObjectSpec.parse("type=tesseract-sponge:level=0.0").toOption.get

    val isFractional4DSpongeMethod = builder.getClass.getDeclaredMethod("isFractional4DSponge", classOf[ObjectSpec])
    isFractional4DSpongeMethod.setAccessible(true)

    isFractional4DSpongeMethod.invoke(builder, spec).asInstanceOf[Boolean] shouldBe false

  it should "handle level 1.0 as integer" in:
    val builder = TriangleMeshSceneBuilder(".")
    val spec = ObjectSpec.parse("type=tesseract-sponge:level=1.0").toOption.get

    val isFractional4DSpongeMethod = builder.getClass.getDeclaredMethod("isFractional4DSponge", classOf[ObjectSpec])
    isFractional4DSpongeMethod.setAccessible(true)

    isFractional4DSpongeMethod.invoke(builder, spec).asInstanceOf[Boolean] shouldBe false

  it should "handle level 2.0 as integer" in:
    val builder = TriangleMeshSceneBuilder(".")
    val spec = ObjectSpec.parse("type=tesseract-sponge:level=2.0").toOption.get

    val isFractional4DSpongeMethod = builder.getClass.getDeclaredMethod("isFractional4DSponge", classOf[ObjectSpec])
    isFractional4DSpongeMethod.setAccessible(true)

    isFractional4DSpongeMethod.invoke(builder, spec).asInstanceOf[Boolean] shouldBe false

  it should "handle very small fractional parts" in:
    val builder = TriangleMeshSceneBuilder(".")
    val spec = ObjectSpec.parse("type=tesseract-sponge:level=1.01").toOption.get

    val isFractional4DSpongeMethod = builder.getClass.getDeclaredMethod("isFractional4DSponge", classOf[ObjectSpec])
    isFractional4DSpongeMethod.setAccessible(true)

    isFractional4DSpongeMethod.invoke(builder, spec).asInstanceOf[Boolean] shouldBe true

  it should "handle fractional part close to 1.0" in:
    val builder = TriangleMeshSceneBuilder(".")
    val spec = ObjectSpec.parse("type=tesseract-sponge:level=1.99").toOption.get

    val isFractional4DSpongeMethod = builder.getClass.getDeclaredMethod("isFractional4DSponge", classOf[ObjectSpec])
    isFractional4DSpongeMethod.setAccessible(true)

    isFractional4DSpongeMethod.invoke(builder, spec).asInstanceOf[Boolean] shouldBe true
    // GPU split: level 1.99 still emits two instances (level 1 + level 2).
    builder.calculateInstanceCount(List(spec)) shouldBe 2L

  // === Validation Tests ===

  "Validation" should "accept fractional level sponges" in:
    val builder = TriangleMeshSceneBuilder(".")
    val specs = List(
      ObjectSpec.parse("type=tesseract-sponge:level=1.5").toOption.get
    )

    val result = builder.validate(specs, maxInstances = 100)
    result shouldBe Right(())

  it should "reject fractional level sponges exceeding instance limit" in:
    val builder = TriangleMeshSceneBuilder(".")
    val specs = List(
      ObjectSpec.parse("type=tesseract-sponge:level=1.5").toOption.get
    )

    // GPU split: a fractional sponge emits two instances (> 0 limit).
    val result = builder.validate(specs, maxInstances = 0)
    result shouldBe a[Left[String, Unit]]
    result.left.getOrElse("") should include("max instances")

  it should "calculate correct instance count with many fractional levels" in:
    val builder = TriangleMeshSceneBuilder(".")
    val specs = (1 to 10).map { i =>
      ObjectSpec.parse(s"type=tesseract-sponge:level=${i}.5").toOption.get
    }.toList

    // GPU split: 10 fractional levels emit two instances each = 20.
    builder.calculateInstanceCount(specs) shouldBe 20L

    val result = builder.validate(specs, maxInstances = 19)
    result shouldBe a[Left[String, Unit]]

    val result2 = builder.validate(specs, maxInstances = 20)
    result2 shouldBe Right(())

  // === Type Classification Tests ===

  "Type classification" should "correctly identify 4D sponges" in:
    ObjectType.is4DSponge("tesseract-sponge") shouldBe true
    ObjectType.is4DSponge("tesseract-sponge-2") shouldBe true
    ObjectType.is4DSponge("tesseract") shouldBe false
    ObjectType.is4DSponge("cube") shouldBe false
    ObjectType.is4DSponge("sponge-volume") shouldBe false
