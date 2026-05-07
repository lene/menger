package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class IcositetrachoronMeshSuite extends AnyFlatSpec with Matchers:

  private def mesh(
      size: Float    = 1.0f,
      center: Vector3 = Vector3(0f, 0f, 0f),
      eyeW: Float    = 3.0f,
      screenW: Float = 1.5f,
      rotXW: Float   = 15f,
      rotYW: Float   = 10f,
      rotZW: Float   = 0f
  ): Mesh4DProjection =
    Mesh4DProjection(Icositetrachoron(size), center, eyeW, screenW, rotXW, rotYW, rotZW)

  "IcositetrachoronMesh" should "generate correct vertex and triangle counts" in:
    val data = mesh().toTriangleMesh
    // 96 triangular faces x 3 vertices = 288 vertices
    data.numVertices shouldBe 288
    // 96 faces x 1 triangle each = 96 triangles
    data.numTriangles shouldBe 96
    data.vertexStride shouldBe 8

  it should "generate valid vertex data (no NaN or Inf)" in:
    val data = mesh().toTriangleMesh
    data.vertices.foreach { v =>
      v.isNaN shouldBe false
      v.isInfinite shouldBe false
    }

  it should "generate valid index data (within vertex bounds)" in:
    val data = mesh().toTriangleMesh
    data.indices.foreach { idx =>
      idx should be >= 0
      idx should be < data.numVertices
    }

  it should "generate normalized normals" in:
    val data = mesh().toTriangleMesh
    for i <- 0 until data.numVertices do
      val nx = data.vertices(i * 8 + 3)
      val ny = data.vertices(i * 8 + 4)
      val nz = data.vertices(i * 8 + 5)
      val length = math.sqrt(nx * nx + ny * ny + nz * nz)
      length shouldBe 1.0 +- 0.01

  it should "apply center translation correctly" in:
    val offset  = Vector3(5f, -3f, 2f)
    val base    = mesh(center = Vector3(0f, 0f, 0f)).toTriangleMesh
    val shifted = mesh(center = offset).toTriangleMesh
    (shifted.vertices(0) - base.vertices(0)) shouldBe offset.x +- 0.001f
    (shifted.vertices(1) - base.vertices(1)) shouldBe offset.y +- 0.001f
    (shifted.vertices(2) - base.vertices(2)) shouldBe offset.z +- 0.001f

  it should "scale geometry correctly" in:
    def xSpan(m: menger.common.TriangleMeshData): Float =
      val xs = (0 until m.numVertices).map(i => m.vertices(i * 8))
      xs.max - xs.min
    xSpan(mesh(size = 2.0f).toTriangleMesh) should be > xSpan(mesh(size = 1.0f).toTriangleMesh)

  it should "produce different geometry with XW rotation" in:
    val base    = mesh(rotXW = 0f, rotYW = 0f, rotZW = 0f).toTriangleMesh
    val rotated = mesh(rotXW = 45f, rotYW = 0f, rotZW = 0f).toTriangleMesh
    base.vertices should not equal rotated.vertices

  it should "produce different geometry with YW rotation" in:
    val base    = mesh(rotXW = 0f, rotYW = 0f, rotZW = 0f).toTriangleMesh
    val rotated = mesh(rotXW = 0f, rotYW = 45f, rotZW = 0f).toTriangleMesh
    base.vertices should not equal rotated.vertices

  it should "produce different geometry with ZW rotation" in:
    val base    = mesh(rotXW = 0f, rotYW = 0f, rotZW = 0f).toTriangleMesh
    val rotated = mesh(rotXW = 0f, rotYW = 0f, rotZW = 45f).toTriangleMesh
    base.vertices should not equal rotated.vertices

  it should "produce same geometry for 0 and 360 degree XW rotation" in:
    val rot0   = mesh(rotXW = 0f, rotYW = 0f, rotZW = 0f).toTriangleMesh
    val rot360 = mesh(rotXW = 360f, rotYW = 0f, rotZW = 0f).toTriangleMesh
    rot0.vertices.zip(rot360.vertices).foreach { case (a, b) =>
      a shouldBe b +- 0.01f
    }

  it should "produce larger projection with closer eye" in:
    def maxExtent(m: menger.common.TriangleMeshData): Float =
      (0 until m.numVertices).map(i => m.vertices(i * 8).abs).max
    maxExtent(mesh(eyeW = 3f,  screenW = 1.5f).toTriangleMesh) should be >
    maxExtent(mesh(eyeW = 10f, screenW = 5f).toTriangleMesh)

  it should "handle edge-case projection distances" in:
    noException should be thrownBy mesh(eyeW = 100f, screenW = 99f).toTriangleMesh

  it should "have visible 4D structure with default rotation" in:
    val m = mesh()
    m.rotXW shouldBe 15f
    m.rotYW shouldBe 10f
    m.toTriangleMesh.numTriangles shouldBe 96

  it should "reject invalid projection parameters (eyeW <= screenW)" in:
    an[IllegalArgumentException] should be thrownBy mesh(eyeW = 1.0f, screenW = 2.0f)

  it should "reject eyeW equal to screenW" in:
    an[IllegalArgumentException] should be thrownBy mesh(eyeW = 2.0f, screenW = 2.0f)

  it should "reject negative projection parameters" in:
    an[IllegalArgumentException] should be thrownBy mesh(eyeW = -1.0f, screenW = 1.0f)

  it should "have correct UV coordinate range" in:
    val data = mesh().toTriangleMesh
    for i <- 0 until data.numVertices do
      val u = data.vertices(i * 8 + 6)
      val v = data.vertices(i * 8 + 7)
      u should (be >= 0f and be <= 1f)
      v should (be >= 0f and be <= 1f)
