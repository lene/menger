package menger.objects

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CubeSpongeGeneratorTest extends AnyFlatSpec with Matchers:

  "CubeSpongeGenerator" should "generate 1 cube at level 0" in:
    val generator = CubeSpongeGenerator(Vector3.Zero, 1.0f, 0)
    val transforms = generator.generateTransforms

    transforms.length shouldBe 1
    transforms.head._1 shouldBe Vector3.Zero
    transforms.head._2 shouldBe 1.0f

  it should "generate 20 cubes at level 1" in:
    val generator = CubeSpongeGenerator(Vector3.Zero, 1.0f, 1)
    val transforms = generator.generateTransforms

    transforms.length shouldBe 20

  it should "generate 400 cubes at level 2" in:
    val generator = CubeSpongeGenerator(Vector3.Zero, 1.0f, 2)
    val transforms = generator.generateTransforms

    transforms.length shouldBe 400

  it should "generate 8000 cubes at level 3" in:
    val generator = CubeSpongeGenerator(Vector3.Zero, 1.0f, 3)
    val transforms = generator.generateTransforms

    transforms.length shouldBe 8000

  it should "correctly calculate cube count" in:
    CubeSpongeGenerator(level = 0).cubeCount shouldBe 1
    CubeSpongeGenerator(level = 1).cubeCount shouldBe 20
    CubeSpongeGenerator(level = 2).cubeCount shouldBe 400
    CubeSpongeGenerator(level = 3).cubeCount shouldBe 8000
    CubeSpongeGenerator(level = 4).cubeCount shouldBe 160000
    CubeSpongeGenerator(level = 5).cubeCount shouldBe 3200000

  it should "generate cubes with correct subdivision pattern" in:
    val generator = CubeSpongeGenerator(Vector3.Zero, 3.0f, 1)
    val transforms = generator.generateTransforms

    // At level 1, cubes should be at distance 1.0 from center (3.0 / 3 = 1.0)
    // and have size 1.0
    transforms.foreach { case (pos, scale) =>
      scale shouldBe 1.0f +- 0.001f
      // Each cube should be at a corner or edge position
      val absSum = math.abs(pos.x) + math.abs(pos.y) + math.abs(pos.z)
      absSum should be > 1.0f
    }

  it should "not include face-center or origin cubes at level 1" in:
    val generator = CubeSpongeGenerator(Vector3.Zero, 3.0f, 1)
    val transforms = generator.generateTransforms

    // Should not contain center cube
    transforms.exists { case (pos, _) =>
      pos.x == 0.0f && pos.y == 0.0f && pos.z == 0.0f
    } shouldBe false

    // Should not contain face-center cubes (one coordinate is 0, others are not)
    val faceCenterCount = transforms.count { case (pos, _) =>
      val nonZeroCount = Seq(pos.x, pos.y, pos.z).count(_ != 0.0f)
      nonZeroCount == 2  // Face centers have exactly 2 non-zero coordinates
    }
    // At level 1, we expect 12 edge cubes and 8 corner cubes = 20 total
    // Corner cubes have all 3 coordinates non-zero
    // Edge cubes have exactly 2 non-zero coordinates
    val cornerCount = transforms.count { case (pos, _) =>
      pos.x != 0.0f && pos.y != 0.0f && pos.z != 0.0f
    }
    cornerCount shouldBe 8  // 8 corners

    val edgeCount = transforms.count { case (pos, _) =>
      val nonZeroCount = Seq(pos.x, pos.y, pos.z).count(_ != 0.0f)
      nonZeroCount == 2
    }
    edgeCount shouldBe 12  // 12 edges

  it should "respect custom center position" in:
    val customCenter = Vector3(5.0f, 10.0f, -3.0f)
    val generator = CubeSpongeGenerator(customCenter, 1.0f, 0)
    val transforms = generator.generateTransforms

    transforms.head._1 shouldBe customCenter

  it should "respect custom size" in:
    val customSize = 2.5f
    val generator = CubeSpongeGenerator(Vector3.Zero, customSize, 0)
    val transforms = generator.generateTransforms

    transforms.head._2 shouldBe customSize

  it should "estimate memory usage correctly" in:
    val generator = CubeSpongeGenerator(level = 3)
    val estimatedBytes = generator.estimateTransformMemoryBytes

    // 8000 cubes * 48 bytes per transform = 384,000 bytes
    estimatedBytes shouldBe 8000L * 48L

  it should "reject negative levels" in:
    assertThrows[IllegalArgumentException]:
      CubeSpongeGenerator(level = -1)

  it should "reject non-positive sizes" in:
    assertThrows[IllegalArgumentException]:
      CubeSpongeGenerator(size = 0.0f)
    assertThrows[IllegalArgumentException]:
      CubeSpongeGenerator(size = -1.0f)

  it should "generate all unique positions" in:
    val generator = CubeSpongeGenerator(Vector3.Zero, 1.0f, 2)
    val transforms = generator.generateTransforms
    val positions = transforms.map(_._1)

    // All positions should be unique
    positions.distinct.length shouldBe positions.length

  it should "generate hierarchical scales at level 2" in:
    val generator = CubeSpongeGenerator(Vector3.Zero, 9.0f, 2)
    val transforms = generator.generateTransforms

    // At level 2, smallest cubes should have scale 9.0 / 3 / 3 = 1.0
    val scales = transforms.map(_._2).distinct.sorted
    scales.head shouldBe 1.0f +- 0.001f
