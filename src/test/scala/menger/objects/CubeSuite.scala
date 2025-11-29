package menger.objects

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Inspectors.forAll

class CubeSuite extends AnyFlatSpec with Matchers:

  "Cube.toTriangleMesh" should "produce 24 vertices (4 per face × 6 faces)" in:
    val mesh = Cube().toTriangleMesh
    mesh.numVertices shouldBe 24

  it should "produce 12 triangles (2 per face × 6 faces)" in:
    val mesh = Cube().toTriangleMesh
    mesh.numTriangles shouldBe 12

  it should "produce 144 floats for vertices (24 × 6)" in:
    val mesh = Cube().toTriangleMesh
    mesh.vertices.length shouldBe 144

  it should "produce 36 indices (12 × 3)" in:
    val mesh = Cube().toTriangleMesh
    mesh.indices.length shouldBe 36

  it should "have all indices in valid range" in:
    val mesh = Cube().toTriangleMesh
    all(mesh.indices) should be >= 0
    all(mesh.indices) should be < mesh.numVertices

  it should "have normalized normals" in:
    val mesh = Cube().toTriangleMesh
    for i <- 0 until mesh.numVertices do
      val nx = mesh.vertices(i * 6 + 3)
      val ny = mesh.vertices(i * 6 + 4)
      val nz = mesh.vertices(i * 6 + 5)
      val length = math.sqrt(nx * nx + ny * ny + nz * nz)
      length shouldBe 1.0 +- 0.001

  it should "have normals pointing in all 6 directions" in:
    val mesh = Cube().toTriangleMesh
    val normals = (0 until mesh.numVertices).map { i =>
      val nx = mesh.vertices(i * 6 + 3)
      val ny = mesh.vertices(i * 6 + 4)
      val nz = mesh.vertices(i * 6 + 5)
      (nx.round, ny.round, nz.round)
    }.toSet

    normals should contain((1, 0, 0))   // +X
    normals should contain((-1, 0, 0))  // -X
    normals should contain((0, 1, 0))   // +Y
    normals should contain((0, -1, 0))  // -Y
    normals should contain((0, 0, 1))   // +Z
    normals should contain((0, 0, -1))  // -Z

  it should "respect center parameter" in:
    val mesh = Cube(center = Vector3(1.0f, 2.0f, 3.0f), scale = 2.0f).toTriangleMesh

    val xs = (0 until mesh.numVertices).map(i => mesh.vertices(i * 6))
    val ys = (0 until mesh.numVertices).map(i => mesh.vertices(i * 6 + 1))
    val zs = (0 until mesh.numVertices).map(i => mesh.vertices(i * 6 + 2))

    // Center at (1, 2, 3), scale 2 means half = 1
    xs.min shouldBe 0.0f +- 0.001f
    xs.max shouldBe 2.0f +- 0.001f
    ys.min shouldBe 1.0f +- 0.001f
    ys.max shouldBe 3.0f +- 0.001f
    zs.min shouldBe 2.0f +- 0.001f
    zs.max shouldBe 4.0f +- 0.001f

  it should "respect scale parameter" in:
    forAll(Seq(0.5f, 1.0f, 2.0f, 10.0f)) { scale =>
      val mesh = Cube(center = Vector3.Zero, scale = scale).toTriangleMesh
      val half = scale / 2

      val xs = (0 until mesh.numVertices).map(i => mesh.vertices(i * 6))
      val ys = (0 until mesh.numVertices).map(i => mesh.vertices(i * 6 + 1))
      val zs = (0 until mesh.numVertices).map(i => mesh.vertices(i * 6 + 2))

      xs.min shouldBe -half +- 0.001f
      xs.max shouldBe half +- 0.001f
      ys.min shouldBe -half +- 0.001f
      ys.max shouldBe half +- 0.001f
      zs.min shouldBe -half +- 0.001f
      zs.max shouldBe half +- 0.001f
    }

  it should "have 4 vertices per normal direction" in:
    val mesh = Cube().toTriangleMesh
    val normalCounts = (0 until mesh.numVertices)
      .map { i =>
        val nx = mesh.vertices(i * 6 + 3).round
        val ny = mesh.vertices(i * 6 + 4).round
        val nz = mesh.vertices(i * 6 + 5).round
        (nx, ny, nz)
      }
      .groupBy(identity)
      .view.mapValues(_.size)
      .toMap

    normalCounts((1, 0, 0)) shouldBe 4
    normalCounts((-1, 0, 0)) shouldBe 4
    normalCounts((0, 1, 0)) shouldBe 4
    normalCounts((0, -1, 0)) shouldBe 4
    normalCounts((0, 0, 1)) shouldBe 4
    normalCounts((0, 0, -1)) shouldBe 4

  "Cube" should "implement TriangleMeshSource trait" in:
    val cube: menger.common.TriangleMeshSource = Cube()
    cube.toTriangleMesh should not be null
