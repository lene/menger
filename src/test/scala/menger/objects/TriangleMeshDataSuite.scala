package menger.objects

import menger.objects.Direction.X
import menger.objects.Direction.Y
import menger.objects.Direction.Z
import org.scalatest.Inspectors.forAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TriangleMeshDataSuite extends AnyFlatSpec with Matchers:

  "TriangleMeshData" should "accept valid vertex and index data" in:
    val vertices = Array(0f, 0f, 0f, 0f, 0f, 1f)
    val indices = Array(0, 0, 0)
    val mesh = TriangleMeshData(vertices, indices)
    mesh.numVertices should be(1)
    mesh.numTriangles should be(1)

  it should "reject vertices not divisible by 6" in:
    val vertices = Array(0f, 0f, 0f, 0f, 0f)
    an[IllegalArgumentException] should be thrownBy:
      TriangleMeshData(vertices, Array.emptyIntArray)

  it should "reject indices not divisible by 3" in:
    an[IllegalArgumentException] should be thrownBy:
      TriangleMeshData(Array.emptyFloatArray, Array(0, 1))

  it should "accept empty arrays" in:
    val mesh = TriangleMeshData(Array.emptyFloatArray, Array.emptyIntArray)
    mesh.numVertices should be(0)
    mesh.numTriangles should be(0)

  "TriangleMeshData.empty" should "have zero vertices and triangles" in:
    TriangleMeshData.empty.numVertices should be(0)
    TriangleMeshData.empty.numTriangles should be(0)

  "TriangleMeshData.merge" should "return empty for empty sequence" in:
    val merged = TriangleMeshData.merge(Seq.empty)
    merged.numVertices should be(0)
    merged.numTriangles should be(0)

  it should "return the same mesh for single-element sequence" in:
    val vertices = Array(0f, 0f, 0f, 0f, 0f, 1f)
    val indices = Array(0, 0, 0)
    val mesh = TriangleMeshData(vertices, indices)
    val merged = TriangleMeshData.merge(Seq(mesh))
    merged.vertices should be(mesh.vertices)
    merged.indices should be(mesh.indices)

  it should "combine vertices from multiple meshes" in:
    val mesh1 = TriangleMeshData(
      Array(0f, 0f, 0f, 0f, 0f, 1f),
      Array(0, 0, 0)
    )
    val mesh2 = TriangleMeshData(
      Array(1f, 1f, 1f, 0f, 0f, 1f),
      Array(0, 0, 0)
    )
    val merged = TriangleMeshData.merge(Seq(mesh1, mesh2))
    merged.numVertices should be(2)
    merged.numTriangles should be(2)

  it should "adjust indices for merged meshes" in:
    val mesh1 = TriangleMeshData(
      Array(0f, 0f, 0f, 0f, 0f, 1f, 1f, 0f, 0f, 0f, 0f, 1f),
      Array(0, 1, 0)
    )
    val mesh2 = TriangleMeshData(
      Array(2f, 0f, 0f, 0f, 0f, 1f, 3f, 0f, 0f, 0f, 0f, 1f),
      Array(0, 1, 0)
    )
    val merged = TriangleMeshData.merge(Seq(mesh1, mesh2))
    merged.indices should be(Array(0, 1, 0, 2, 3, 2))

  "Face.toTriangleMesh" should "produce 4 vertices and 2 triangles" in:
    forAll(Seq(X, Y, Z, -X, -Y, -Z)) { normal =>
      val face = Face(0, 0, 0, 1, normal)
      val mesh = face.toTriangleMesh
      mesh.numVertices should be(4)
      mesh.numTriangles should be(2)
    }

  it should "have vertices at correct positions for Z normal" in:
    val face = Face(0, 0, 0, 1, Z)
    val mesh = face.toTriangleMesh
    val positions = (0 until 4).map { i =>
      (mesh.vertices(i * 6), mesh.vertices(i * 6 + 1), mesh.vertices(i * 6 + 2))
    }
    positions should contain((-0.5f, -0.5f, 0f))
    positions should contain((0.5f, -0.5f, 0f))
    positions should contain((0.5f, 0.5f, 0f))
    positions should contain((-0.5f, 0.5f, 0f))

  it should "have correct normals for Z normal" in:
    val face = Face(0, 0, 0, 1, Z)
    val mesh = face.toTriangleMesh
    for i <- 0 until 4 do
      mesh.vertices(i * 6 + 3) should be(0f)
      mesh.vertices(i * 6 + 4) should be(0f)
      mesh.vertices(i * 6 + 5) should be(1f)

  it should "have correct normals for -Z normal" in:
    val face = Face(0, 0, 0, 1, -Z)
    val mesh = face.toTriangleMesh
    for i <- 0 until 4 do
      mesh.vertices(i * 6 + 3) should be(0f)
      mesh.vertices(i * 6 + 4) should be(0f)
      mesh.vertices(i * 6 + 5) should be(-1f)

  it should "have triangle indices referencing valid vertices" in:
    forAll(Seq(X, Y, Z, -X, -Y, -Z)) { normal =>
      val face = Face(0, 0, 0, 1, normal)
      val mesh = face.toTriangleMesh
      for idx <- mesh.indices do
        idx should be >= 0
        idx should be < mesh.numVertices
    }

  it should "respect face center position" in:
    val face = Face(1, 2, 3, 1, Z)
    val mesh = face.toTriangleMesh
    val positions = (0 until 4).map { i =>
      (mesh.vertices(i * 6), mesh.vertices(i * 6 + 1), mesh.vertices(i * 6 + 2))
    }
    positions should contain((0.5f, 1.5f, 3f))
    positions should contain((1.5f, 1.5f, 3f))
    positions should contain((1.5f, 2.5f, 3f))
    positions should contain((0.5f, 2.5f, 3f))

  it should "respect face scale" in:
    val face = Face(0, 0, 0, 2, Z)
    val mesh = face.toTriangleMesh
    val positions = (0 until 4).map { i =>
      (mesh.vertices(i * 6), mesh.vertices(i * 6 + 1), mesh.vertices(i * 6 + 2))
    }
    positions should contain((-1f, -1f, 0f))
    positions should contain((1f, -1f, 0f))
    positions should contain((1f, 1f, 0f))
    positions should contain((-1f, 1f, 0f))

  "merging Face meshes" should "produce correct combined mesh" in:
    val faces = Seq(
      Face(0, 0, 0, 1, Z),
      Face(0, 0, 1, 1, Z)
    )
    val meshes = faces.map(_.toTriangleMesh)
    val merged = TriangleMeshData.merge(meshes)
    merged.numVertices should be(8)
    merged.numTriangles should be(4)

  "Direction.toFloatArray" should "return correct values" in:
    X.toFloatArray should be(Array(1f, 0f, 0f))
    Y.toFloatArray should be(Array(0f, 1f, 0f))
    Z.toFloatArray should be(Array(0f, 0f, 1f))
    (-X).toFloatArray should be(Array(-1f, 0f, 0f))
    (-Y).toFloatArray should be(Array(0f, -1f, 0f))
    (-Z).toFloatArray should be(Array(0f, 0f, -1f))
