package menger.objects

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.tagobjects.Slow

class SpongeBySurfaceMeshSuite extends AnyFlatSpec with Matchers:

  given menger.ProfilingConfig = menger.ProfilingConfig.disabled

  "Face.toTriangleMesh" should "generate a quad mesh with 2 triangles" in:
    val face = Face(0, 0, 0, 1.0f, Direction.Z)
    val mesh = face.toTriangleMesh
    mesh.numTriangles shouldBe 2
    mesh.numVertices shouldBe 4

  "SpongeBySurface.toTriangleMesh" should "generate 12 triangles at level 0 (6 faces * 2 triangles)" in:
    val sponge = SpongeBySurface(Vector3.Zero, 1.0f, level = 0)
    val mesh = sponge.toTriangleMesh
    mesh.numTriangles shouldBe 12
    mesh.numVertices shouldBe 24

  it should "generate correct number of triangles at level 1" in:
    val sponge = SpongeBySurface(Vector3.Zero, 1.0f, level = 1)
    val mesh = sponge.toTriangleMesh
    // Level 1: 6 faces * (8 unrotated + 4 rotated) = 6 * 12 = 72 sub-faces
    // Each sub-face = 2 triangles, so 72 * 2 = 144 triangles
    mesh.numTriangles shouldBe 144
    mesh.numVertices shouldBe 288

  it should "generate correct number of triangles at level 2" taggedAs Slow in:
    val sponge = SpongeBySurface(Vector3.Zero, 1.0f, level = 2)
    val mesh = sponge.toTriangleMesh
    // Level 2: more complex subdivision
    // At level 1 we have 12 sub-faces per original face
    // At level 2, each of those 12 subdivides into 12 more = 144 per original face
    // 6 original faces * 144 = 864 sub-faces
    // 864 * 2 triangles = 1728 triangles
    mesh.numTriangles shouldBe 1728
    mesh.numVertices shouldBe 3456

  it should "position faces correctly at level 0" in:
    val sponge = SpongeBySurface(Vector3.Zero, 2.0f, level = 0)
    val mesh = sponge.toTriangleMesh

    val vertices = mesh.vertices
    val xs = (0 until mesh.numVertices).map(i => vertices(i * 6))
    val ys = (0 until mesh.numVertices).map(i => vertices(i * 6 + 1))
    val zs = (0 until mesh.numVertices).map(i => vertices(i * 6 + 2))

    xs.min should be < -0.9f
    xs.max should be > 0.9f
    ys.min should be < -0.9f
    ys.max should be > 0.9f
    zs.min should be < -0.9f
    zs.max should be > 0.9f

  it should "preserve proper normals for faces" in:
    val sponge = SpongeBySurface(Vector3.Zero, 1.0f, level = 0)
    val mesh = sponge.toTriangleMesh

    for i <- 0 until mesh.numVertices do
      val nx = mesh.vertices(i * 6 + 3)
      val ny = mesh.vertices(i * 6 + 4)
      val nz = mesh.vertices(i * 6 + 5)
      val lengthSquared = nx * nx + ny * ny + nz * nz
      lengthSquared shouldBe 1.0f +- 0.01f
