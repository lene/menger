package menger.engines.scene

import menger.ObjectSpec
import menger.ProfilingConfig
import menger.common.Color
import menger.common.ObjectType
import menger.optix.Material
import menger.optix.OptiXRenderer
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

  it should "count 2 for fractional level sponges" in:
    val builder = TriangleMeshSceneBuilder(".")
    val specs = List(
      ObjectSpec.parse("type=tesseract-sponge:level=1.5").toOption.get
    )

    builder.calculateInstanceCount(specs) shouldBe 2L

  it should "count correctly for mixed integer and fractional levels" in:
    val builder = TriangleMeshSceneBuilder(".")
    val specs = List(
      ObjectSpec.parse("type=tesseract-sponge:level=1").toOption.get,
      ObjectSpec.parse("type=tesseract-sponge:level=1.5").toOption.get,
      ObjectSpec.parse("type=tesseract-sponge:level=2").toOption.get,
      ObjectSpec.parse("type=tesseract-sponge:level=2.3").toOption.get
    )

    // 1 + 2 + 1 + 2 = 6
    builder.calculateInstanceCount(specs) shouldBe 6L

  it should "count correctly for multiple fractional levels" in:
    val builder = TriangleMeshSceneBuilder(".")
    val specs = List(
      ObjectSpec.parse("type=tesseract-sponge:level=1.2").toOption.get,
      ObjectSpec.parse("type=tesseract-sponge:level=1.5").toOption.get,
      ObjectSpec.parse("type=tesseract-sponge:level=1.8").toOption.get
    )

    // Each fractional level creates 2 instances
    builder.calculateInstanceCount(specs) shouldBe 6L

  it should "count correctly for non-sponge objects" in:
    val builder = TriangleMeshSceneBuilder(".")
    val specs = List(
      ObjectSpec.parse("type=cube").toOption.get,
      ObjectSpec.parse("type=tesseract").toOption.get
    )

    builder.calculateInstanceCount(specs) shouldBe 2L

  // === Geometry Grouping Tests ===

  "Geometry grouping" should "create single group for integer level" in:
    val builder = TriangleMeshSceneBuilder(".")
    val spec = ObjectSpec.parse("type=tesseract-sponge:level=1:pos=1,2,3").toOption.get
    val specs = List(spec)

    val groupByGeometryMethod = builder.getClass.getDeclaredMethod("groupByGeometry", classOf[List[ObjectSpec]])
    groupByGeometryMethod.setAccessible(true)

    val groups = groupByGeometryMethod.invoke(builder, specs).asInstanceOf[Map[ObjectSpec, List[(ObjectSpec, Float)]]]

    groups.size shouldBe 1
    val (geomSpec, instances) = groups.head
    geomSpec.level shouldBe Some(1.0f)
    instances.length shouldBe 1
    instances.head._1 shouldBe spec
    instances.head._2 shouldBe 1.0f

  it should "create two groups for fractional level" in:
    val builder = TriangleMeshSceneBuilder(".")
    val spec = ObjectSpec.parse("type=tesseract-sponge:level=1.5:pos=1,2,3").toOption.get
    val specs = List(spec)

    val groupByGeometryMethod = builder.getClass.getDeclaredMethod("groupByGeometry", classOf[List[ObjectSpec]])
    groupByGeometryMethod.setAccessible(true)

    val groups = groupByGeometryMethod.invoke(builder, specs).asInstanceOf[Map[ObjectSpec, List[(ObjectSpec, Float)]]]

    groups.size shouldBe 2

    // Should have groups for level 1 and level 2
    val levels = groups.keys.flatMap(_.level).toSet
    levels shouldBe Set(1.0f, 2.0f)

    // Level 2 (next level) should have alpha = 1.0
    val level2Group = groups.find(_._1.level.contains(2.0f)).get
    level2Group._2.length shouldBe 1
    level2Group._2.head._2 shouldBe 1.0f

    // Level 1 (current level) should have alpha = 0.5 (1.0 - 0.5)
    val level1Group = groups.find(_._1.level.contains(1.0f)).get
    level1Group._2.length shouldBe 1
    level1Group._2.head._2 shouldBe 0.5f

  it should "calculate correct alpha for various fractional parts" in:
    val builder = TriangleMeshSceneBuilder(".")

    val testCases = List(
      (1.25f, 0.75f),  // fractional = 0.25, alpha = 1 - 0.25 = 0.75
      (1.5f, 0.5f),    // fractional = 0.5, alpha = 1 - 0.5 = 0.5
      (1.75f, 0.25f),  // fractional = 0.75, alpha = 1 - 0.75 = 0.25
      (2.1f, 0.9f),    // fractional = 0.1, alpha = 1 - 0.1 = 0.9
      (2.9f, 0.1f)     // fractional = 0.9, alpha = 1 - 0.9 = 0.1
    )

    val groupByGeometryMethod = builder.getClass.getDeclaredMethod("groupByGeometry", classOf[List[ObjectSpec]])
    groupByGeometryMethod.setAccessible(true)

    testCases.foreach { case (level, expectedAlpha) =>
      val spec = ObjectSpec.parse(s"type=tesseract-sponge:level=$level").toOption.get
      val groups = groupByGeometryMethod.invoke(builder, List(spec)).asInstanceOf[Map[ObjectSpec, List[(ObjectSpec, Float)]]]

      // Find the current level group (floor(level))
      val currentLevelGroup = groups.find(_._1.level.contains(level.floor)).get
      val actualAlpha = currentLevelGroup._2.head._2

      actualAlpha shouldBe expectedAlpha +- 0.01f
    }

  it should "merge multiple specs with same integer level" in:
    val builder = TriangleMeshSceneBuilder(".")
    val specs = List(
      ObjectSpec.parse("type=tesseract-sponge:level=1:pos=0,0,0").toOption.get,
      ObjectSpec.parse("type=tesseract-sponge:level=1:pos=1,0,0").toOption.get,
      ObjectSpec.parse("type=tesseract-sponge:level=1:pos=2,0,0").toOption.get
    )

    val groupByGeometryMethod = builder.getClass.getDeclaredMethod("groupByGeometry", classOf[List[ObjectSpec]])
    groupByGeometryMethod.setAccessible(true)

    val groups = groupByGeometryMethod.invoke(builder, specs).asInstanceOf[Map[ObjectSpec, List[(ObjectSpec, Float)]]]

    groups.size shouldBe 1
    val instances = groups.head._2
    instances.length shouldBe 3

  it should "handle mixed integer and fractional levels" in:
    val builder = TriangleMeshSceneBuilder(".")
    val specs = List(
      ObjectSpec.parse("type=tesseract-sponge:level=1").toOption.get,
      ObjectSpec.parse("type=tesseract-sponge:level=1.5").toOption.get
    )

    val groupByGeometryMethod = builder.getClass.getDeclaredMethod("groupByGeometry", classOf[List[ObjectSpec]])
    groupByGeometryMethod.setAccessible(true)

    val groups = groupByGeometryMethod.invoke(builder, specs).asInstanceOf[Map[ObjectSpec, List[(ObjectSpec, Float)]]]

    // Should have groups for level 1 and level 2
    // Level 1 group should have: integer spec (alpha=1.0) + fractional current (alpha=0.5)
    // Level 2 group should have: fractional next (alpha=1.0)
    groups.size shouldBe 2

    val level1Group = groups.find(_._1.level.contains(1.0f)).get
    level1Group._2.length shouldBe 2  // integer + fractional current

    val level2Group = groups.find(_._1.level.contains(2.0f)).get
    level2Group._2.length shouldBe 1  // fractional next

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

    val groupByGeometryMethod = builder.getClass.getDeclaredMethod("groupByGeometry", classOf[List[ObjectSpec]])
    groupByGeometryMethod.setAccessible(true)

    val groups = groupByGeometryMethod.invoke(builder, List(spec)).asInstanceOf[Map[ObjectSpec, List[(ObjectSpec, Float)]]]

    // Should have level 1 (alpha=0.01) and level 2 (alpha=1.0)
    groups.size shouldBe 2
    val level1Group = groups.find(_._1.level.contains(1.0f)).get
    level1Group._2.head._2 shouldBe 0.01f +- 0.01f

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

    // Fractional level creates 2 instances, so max of 1 should fail
    val result = builder.validate(specs, maxInstances = 1)
    result shouldBe a[Left[String, Unit]]
    result.left.getOrElse("") should include("max instances")

  it should "calculate correct instance count with many fractional levels" in:
    val builder = TriangleMeshSceneBuilder(".")
    val specs = (1 to 10).map { i =>
      ObjectSpec.parse(s"type=tesseract-sponge:level=${i}.5").toOption.get
    }.toList

    // 10 fractional levels = 20 instances
    builder.calculateInstanceCount(specs) shouldBe 20L

    val result = builder.validate(specs, maxInstances = 19)
    result shouldBe a[Left[String, Unit]]

    val result2 = builder.validate(specs, maxInstances = 20)
    result2 shouldBe Right(())

  // === Alpha Calculation Tests ===

  "Alpha calculation" should "match LibGDX FractionalRotatedProjection formula" in:
    // LibGDX formula: alpha = 1.0 - fractionalPart
    val testCases = Map(
      1.0f -> 1.0f,   // Integer level -> opaque
      1.25f -> 0.75f, // 25% -> 75% opaque
      1.5f -> 0.5f,   // 50% -> 50% opaque
      1.75f -> 0.25f, // 75% -> 25% opaque
      2.1f -> 0.9f,   // 10% -> 90% opaque
      2.99f -> 0.01f  // 99% -> 1% opaque
    )

    val builder = TriangleMeshSceneBuilder(".")
    val groupByGeometryMethod = builder.getClass.getDeclaredMethod("groupByGeometry", classOf[List[ObjectSpec]])
    groupByGeometryMethod.setAccessible(true)

    testCases.foreach { case (level, expectedCurrentAlpha) =>
      val spec = ObjectSpec.parse(s"type=tesseract-sponge:level=$level").toOption.get
      val groups = groupByGeometryMethod.invoke(builder, List(spec)).asInstanceOf[Map[ObjectSpec, List[(ObjectSpec, Float)]]]

      if level == level.floor then
        // Integer level: single group with alpha=1.0
        groups.size shouldBe 1
        groups.head._2.head._2 shouldBe 1.0f
      else
        // Fractional level: two groups
        groups.size shouldBe 2

        // Current level should have calculated alpha
        val currentLevelGroup = groups.find(_._1.level.contains(level.floor)).get
        currentLevelGroup._2.head._2 shouldBe expectedCurrentAlpha +- 0.01f

        // Next level should be fully opaque
        val nextLevelGroup = groups.find(_._1.level.contains((level + 1).floor)).get
        nextLevelGroup._2.head._2 shouldBe 1.0f
    }

  // === Type Classification Tests ===

  "Type classification" should "correctly identify 4D sponges" in:
    ObjectType.is4DSponge("tesseract-sponge") shouldBe true
    ObjectType.is4DSponge("tesseract-sponge-2") shouldBe true
    ObjectType.is4DSponge("tesseract") shouldBe false
    ObjectType.is4DSponge("cube") shouldBe false
    ObjectType.is4DSponge("sponge-volume") shouldBe false
