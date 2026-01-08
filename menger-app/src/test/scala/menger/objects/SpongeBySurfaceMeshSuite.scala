package menger.objects

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.tagobjects.Slow

class SpongeBySurfaceMeshSuite extends AnyFlatSpec with Matchers:

  given menger.ProfilingConfig = menger.ProfilingConfig.disabled

  private val QUAD_TRIANGLE_COUNT = 2
  private val QUAD_VERTEX_COUNT = 4

  private val LEVEL_0_TRIANGLES = 12
  private val LEVEL_0_VERTICES = 24
  private val LEVEL_1_TRIANGLES = 144
  private val LEVEL_1_VERTICES = 288
  private val LEVEL_2_TRIANGLES = 1728
  private val LEVEL_2_VERTICES = 3456

  "Face.toTriangleMesh" should "generate a quad mesh with 2 triangles" in:
    val face = Face(0, 0, 0, 1.0f, Direction.Z)
    val mesh = face.toTriangleMesh
    mesh.numTriangles shouldBe QUAD_TRIANGLE_COUNT
    mesh.numVertices shouldBe QUAD_VERTEX_COUNT

  "SpongeBySurface.toTriangleMesh" should "generate 12 triangles at level 0 (6 faces * 2 triangles)" in:
    val sponge = SpongeBySurface(Vector3.Zero, 1.0f, level = 0)
    val mesh = sponge.toTriangleMesh
    mesh.numTriangles shouldBe LEVEL_0_TRIANGLES
    mesh.numVertices shouldBe LEVEL_0_VERTICES

  it should "generate correct number of triangles at level 1" in:
    val sponge = SpongeBySurface(Vector3.Zero, 1.0f, level = 1)
    val mesh = sponge.toTriangleMesh
    mesh.numTriangles shouldBe LEVEL_1_TRIANGLES
    mesh.numVertices shouldBe LEVEL_1_VERTICES

  it should "generate correct number of triangles at level 2" taggedAs Slow in:
    val sponge = SpongeBySurface(Vector3.Zero, 1.0f, level = 2)
    val mesh = sponge.toTriangleMesh
    mesh.numTriangles shouldBe LEVEL_2_TRIANGLES
    mesh.numVertices shouldBe LEVEL_2_VERTICES

  it should "position faces correctly at level 0" in:
    val sponge = SpongeBySurface(Vector3.Zero, 2.0f, level = 0)
    val mesh = sponge.toTriangleMesh
    val stride = mesh.vertexStride

    val vertices = mesh.vertices
    val xs = (0 until mesh.numVertices).map(i => vertices(i * stride))
    val ys = (0 until mesh.numVertices).map(i => vertices(i * stride + 1))
    val zs = (0 until mesh.numVertices).map(i => vertices(i * stride + 2))

    xs.min should be < -0.9f
    xs.max should be > 0.9f
    ys.min should be < -0.9f
    ys.max should be > 0.9f
    zs.min should be < -0.9f
    zs.max should be > 0.9f

  it should "preserve proper normals for faces" in:
    val sponge = SpongeBySurface(Vector3.Zero, 1.0f, level = 0)
    val mesh = sponge.toTriangleMesh
    val stride = mesh.vertexStride

    for i <- 0 until mesh.numVertices do
      val nx = mesh.vertices(i * stride + 3)
      val ny = mesh.vertices(i * stride + 4)
      val nz = mesh.vertices(i * stride + 5)
      val lengthSquared = nx * nx + ny * ny + nz * nz
      lengthSquared shouldBe 1.0f +- 0.01f
