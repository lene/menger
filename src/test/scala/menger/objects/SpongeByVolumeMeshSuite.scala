package menger.objects

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.tagobjects.Slow

class SpongeByVolumeMeshSuite extends AnyFlatSpec with Matchers:

  "Cube.toTriangleMesh" should "generate a cube mesh with 12 triangles" in:
    val cube = Cube(Vector3.Zero, 1.0f)
    val mesh = cube.toTriangleMesh
    mesh.numTriangles shouldBe 12
    mesh.numVertices shouldBe 24

  "SpongeByVolume.toTriangleMesh" should "generate a single cube at level 0" in:
    val sponge = SpongeByVolume(Vector3.Zero, 1.0f, level = 0)
    val mesh = sponge.toTriangleMesh
    mesh.numTriangles shouldBe 12
    mesh.numVertices shouldBe 24

  it should "generate 20 cubes at level 1 (240 triangles)" in:
    val sponge = SpongeByVolume(Vector3.Zero, 1.0f, level = 1)
    val mesh = sponge.toTriangleMesh
    mesh.numTriangles shouldBe 240
    mesh.numVertices shouldBe 480

  it should "generate 400 cubes at level 2 (4800 triangles)" taggedAs Slow in:
    val sponge = SpongeByVolume(Vector3.Zero, 1.0f, level = 2)
    val mesh = sponge.toTriangleMesh
    mesh.numTriangles shouldBe 4800
    mesh.numVertices shouldBe 9600

  it should "position cubes correctly at level 1" in:
    val sponge = SpongeByVolume(Vector3.Zero, 3.0f, level = 1)
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
    val sponge = SpongeByVolume(Vector3.Zero, 1.0f, level = 0)
    val mesh = sponge.toTriangleMesh

    for i <- 0 until mesh.numVertices do
      val nx = mesh.vertices(i * 6 + 3)
      val ny = mesh.vertices(i * 6 + 4)
      val nz = mesh.vertices(i * 6 + 5)
      val lengthSquared = nx * nx + ny * ny + nz * nz
      lengthSquared shouldBe 1.0f +- 0.01f
