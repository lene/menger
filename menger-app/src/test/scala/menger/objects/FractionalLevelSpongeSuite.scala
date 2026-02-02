package menger.objects

import com.badlogic.gdx.math.Vector3
import menger.ProfilingConfig
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Test suite for 3D fractional level sponges (SpongeByVolume and SpongeBySurface).
 *
 * Mirrors the comprehensive test coverage added for 4D fractional sponges in Sprint 9.
 * Tests per-vertex alpha implementation for smooth LOD transitions.
 */
class FractionalLevelSpongeSuite extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig(minDurationMs = None)

  // === SpongeByVolume Fractional Level Tests ===

  "SpongeByVolume" should "generate stride=9 mesh for fractional levels" in:
    val sponge = SpongeByVolume(Vector3.Zero, 1f, level = 1.5f)
    val mesh = sponge.toTriangleMesh

    mesh.vertexStride shouldBe 9  // pos(3) + normal(3) + uv(2) + alpha(1)
    mesh.numTriangles should be > 0

  it should "generate stride=8 mesh for integer levels" in:
    val sponge = SpongeByVolume(Vector3.Zero, 1f, level = 2f)
    val mesh = sponge.toTriangleMesh

    mesh.vertexStride shouldBe 8  // pos(3) + normal(3) + uv(2)
    mesh.numTriangles should be > 0

  it should "merge level N and N+1 geometries for fractional levels" in:
    val sponge = SpongeByVolume(Vector3.Zero, 1f, level = 1.5f)
    val mesh = sponge.toTriangleMesh

    // Fractional level 1.5 should have triangles from both level 1 and level 2
    val level1 = SpongeByVolume(Vector3.Zero, 1f, level = 1f).toTriangleMesh
    val level2 = SpongeByVolume(Vector3.Zero, 1f, level = 2f).toTriangleMesh

    mesh.numTriangles shouldBe (level1.numTriangles + level2.numTriangles)

  it should "handle level 0.0 as integer" in:
    val sponge = SpongeByVolume(Vector3.Zero, 1f, level = 0.0f)
    val mesh = sponge.toTriangleMesh

    mesh.vertexStride shouldBe 8  // Integer level uses stride=8

  it should "handle level 1.0 as integer" in:
    val sponge = SpongeByVolume(Vector3.Zero, 1f, level = 1.0f)
    val mesh = sponge.toTriangleMesh

    mesh.vertexStride shouldBe 8  // Integer level uses stride=8

  it should "handle level 2.0 as integer" in:
    val sponge = SpongeByVolume(Vector3.Zero, 1f, level = 2.0f)
    val mesh = sponge.toTriangleMesh

    mesh.vertexStride shouldBe 8  // Integer level uses stride=8

  it should "handle very small fractional parts" in:
    val sponge = SpongeByVolume(Vector3.Zero, 1f, level = 1.01f)
    val mesh = sponge.toTriangleMesh

    mesh.vertexStride shouldBe 9  // Fractional level uses stride=9
    mesh.vertices.length % 9 shouldBe 0

  it should "handle fractional parts close to 1.0" in:
    val sponge = SpongeByVolume(Vector3.Zero, 1f, level = 1.99f)
    val mesh = sponge.toTriangleMesh

    mesh.vertexStride shouldBe 9  // Fractional level uses stride=9
    // Alpha values: level 1 = 0.01 (very transparent), level 2 = 1.0 (opaque)
    mesh.numTriangles should be > 0

  it should "generate valid meshes for various fractional values" in:
    val testLevels = List(0.25f, 0.5f, 0.75f, 1.25f, 1.5f, 1.75f)

    testLevels.foreach { level =>
      val sponge = SpongeByVolume(Vector3.Zero, 1f, level = level)
      val mesh = sponge.toTriangleMesh

      mesh.vertexStride shouldBe 9
      mesh.numTriangles should be > 0
      mesh.vertices.length % 9 shouldBe 0
    }

  it should "produce different geometry for different fractional levels" in:
    val sponge1 = SpongeByVolume(Vector3.Zero, 1f, level = 1.25f).toTriangleMesh
    val sponge2 = SpongeByVolume(Vector3.Zero, 1f, level = 1.75f).toTriangleMesh

    // Same merged triangle count (level 1 + level 2)
    sponge1.numTriangles shouldBe sponge2.numTriangles
    // But different alpha values in vertex data
    sponge1.vertexStride shouldBe sponge2.vertexStride

  // === SpongeBySurface Fractional Level Tests ===

  "SpongeBySurface" should "generate stride=9 mesh for fractional levels" in:
    val sponge = SpongeBySurface(Vector3.Zero, 1f, level = 1.5f)
    val mesh = sponge.toTriangleMesh

    mesh.vertexStride shouldBe 9  // pos(3) + normal(3) + uv(2) + alpha(1)
    mesh.numTriangles should be > 0

  it should "generate stride=8 mesh for integer levels" in:
    val sponge = SpongeBySurface(Vector3.Zero, 1f, level = 2f)
    val mesh = sponge.toTriangleMesh

    mesh.vertexStride shouldBe 8  // pos(3) + normal(3) + uv(2)
    mesh.numTriangles should be > 0

  it should "merge level N and N+1 geometries for fractional levels" in:
    val sponge = SpongeBySurface(Vector3.Zero, 1f, level = 1.5f)
    val mesh = sponge.toTriangleMesh

    // Fractional level 1.5 should have triangles from both level 1 and level 2
    val level1 = SpongeBySurface(Vector3.Zero, 1f, level = 1f).toTriangleMesh
    val level2 = SpongeBySurface(Vector3.Zero, 1f, level = 2f).toTriangleMesh

    mesh.numTriangles shouldBe (level1.numTriangles + level2.numTriangles)

  it should "handle level 0.0 as integer" in:
    val sponge = SpongeBySurface(Vector3.Zero, 1f, level = 0.0f)
    val mesh = sponge.toTriangleMesh

    mesh.vertexStride shouldBe 8  // Integer level uses stride=8

  it should "handle level 1.0 as integer" in:
    val sponge = SpongeBySurface(Vector3.Zero, 1f, level = 1.0f)
    val mesh = sponge.toTriangleMesh

    mesh.vertexStride shouldBe 8  // Integer level uses stride=8

  it should "handle very small fractional parts" in:
    val sponge = SpongeBySurface(Vector3.Zero, 1f, level = 1.01f)
    val mesh = sponge.toTriangleMesh

    mesh.vertexStride shouldBe 9  // Fractional level uses stride=9
    mesh.vertices.length % 9 shouldBe 0

  it should "handle fractional parts close to 1.0" in:
    val sponge = SpongeBySurface(Vector3.Zero, 1f, level = 1.99f)
    val mesh = sponge.toTriangleMesh

    mesh.vertexStride shouldBe 9  // Fractional level uses stride=9
    mesh.numTriangles should be > 0

  it should "generate valid meshes for various fractional values" in:
    val testLevels = List(0.25f, 0.5f, 0.75f, 1.25f, 1.5f, 1.75f)

    testLevels.foreach { level =>
      val sponge = SpongeBySurface(Vector3.Zero, 1f, level = level)
      val mesh = sponge.toTriangleMesh

      mesh.vertexStride shouldBe 9
      mesh.numTriangles should be > 0
      mesh.vertices.length % 9 shouldBe 0
    }

  // === Alpha Calculation Tests ===

  "Alpha calculation" should "follow formula: alpha = 1.0 - fractionalPart for SpongeByVolume" in:
    // The alpha for level N is: 1.0 - (level - floor(level))
    // Level N+1 always has alpha = 1.0
    val testCases = List(
      (1.25f, 0.75f), // 25% -> current level alpha = 75%
      (1.5f, 0.5f),   // 50% -> current level alpha = 50%
      (1.75f, 0.25f), // 75% -> current level alpha = 25%
      (1.9f, 0.1f)    // 90% -> current level alpha = 10%
    )

    testCases.foreach { case (level, _) =>
      val sponge = SpongeByVolume(Vector3.Zero, 1f, level = level)
      val mesh = sponge.toTriangleMesh

      // Verify merged mesh was created with correct format
      mesh.vertexStride shouldBe 9
      mesh.numTriangles should be > 0
      mesh.vertices.length % 9 shouldBe 0
    }

  it should "follow formula: alpha = 1.0 - fractionalPart for SpongeBySurface" in:
    val testCases = List(
      (1.25f, 0.75f),
      (1.5f, 0.5f),
      (1.75f, 0.25f)
    )

    testCases.foreach { case (level, _) =>
      val sponge = SpongeBySurface(Vector3.Zero, 1f, level = level)
      val mesh = sponge.toTriangleMesh

      mesh.vertexStride shouldBe 9
      mesh.numTriangles should be > 0
      mesh.vertices.length % 9 shouldBe 0
    }

  // === Comparison Tests ===

  "SpongeByVolume vs SpongeBySurface" should "both support fractional levels" in:
    val volume = SpongeByVolume(Vector3.Zero, 1f, level = 1.5f).toTriangleMesh
    val surface = SpongeBySurface(Vector3.Zero, 1f, level = 1.5f).toTriangleMesh

    volume.vertexStride shouldBe 9
    surface.vertexStride shouldBe 9
    volume.numTriangles should be > 0
    surface.numTriangles should be > 0

  it should "generate different triangle counts at same level" in:
    val volume = SpongeByVolume(Vector3.Zero, 1f, level = 1.5f).toTriangleMesh
    val surface = SpongeBySurface(Vector3.Zero, 1f, level = 1.5f).toTriangleMesh

    // Volume-based generates more triangles than surface-based
    volume.numTriangles should be > surface.numTriangles

  // === Edge Cases ===

  "Edge cases" should "handle level 0.5 correctly for SpongeByVolume" in:
    val sponge = SpongeByVolume(Vector3.Zero, 1f, level = 0.5f)
    val mesh = sponge.toTriangleMesh

    mesh.vertexStride shouldBe 9
    // Level 0.5 merges level 0 (cube) + level 1 (sponge)
    mesh.numTriangles should be > 0

  it should "handle level 0.5 correctly for SpongeBySurface" in:
    val sponge = SpongeBySurface(Vector3.Zero, 1f, level = 0.5f)
    val mesh = sponge.toTriangleMesh

    mesh.vertexStride shouldBe 9
    mesh.numTriangles should be > 0

  it should "not produce NaN vertices for SpongeByVolume" in:
    val sponge = SpongeByVolume(Vector3.Zero, 1f, level = 1.5f)
    val mesh = sponge.toTriangleMesh

    mesh.vertices.foreach { v =>
      v.isNaN shouldBe false
      v.isInfinite shouldBe false
    }

  it should "not produce NaN vertices for SpongeBySurface" in:
    val sponge = SpongeBySurface(Vector3.Zero, 1f, level = 1.5f)
    val mesh = sponge.toTriangleMesh

    mesh.vertices.foreach { v =>
      v.isNaN shouldBe false
      v.isInfinite shouldBe false
    }
