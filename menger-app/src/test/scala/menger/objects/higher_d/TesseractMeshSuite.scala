package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TesseractMeshSuite extends AnyFlatSpec with Matchers:

  "TesseractMesh" should "generate correct vertex and triangle counts" in:
    val mesh = TesseractMesh()
    val data = mesh.toTriangleMesh

    // 24 faces x 4 vertices = 96 vertices
    data.numVertices shouldBe 96
    // 24 faces x 2 triangles = 48 triangles
    data.numTriangles shouldBe 48
    data.vertexStride shouldBe 8

  it should "generate valid vertex data (no NaN or Inf)" in:
    val mesh = TesseractMesh()
    val data = mesh.toTriangleMesh

    data.vertices.foreach { v =>
      v.isNaN shouldBe false
      v.isInfinite shouldBe false
    }

  it should "generate valid index data (within vertex bounds)" in:
    val mesh = TesseractMesh()
    val data = mesh.toTriangleMesh

    data.indices.foreach { idx =>
      idx should be >= 0
      idx should be < data.numVertices
    }

  it should "generate normalized normals" in:
    val mesh = TesseractMesh()
    val data = mesh.toTriangleMesh

    for i <- 0 until data.numVertices do
      val nx = data.vertices(i * 8 + 3)
      val ny = data.vertices(i * 8 + 4)
      val nz = data.vertices(i * 8 + 5)
      val length = math.sqrt(nx * nx + ny * ny + nz * nz)
      length shouldBe 1.0 +- 0.01

  it should "apply center translation correctly" in:
    val offset = Vector3(5f, -3f, 2f)
    val centered = TesseractMesh(center = Vector3(0f, 0f, 0f)).toTriangleMesh
    val translated = TesseractMesh(center = offset).toTriangleMesh

    val dx = translated.vertices(0) - centered.vertices(0)
    val dy = translated.vertices(1) - centered.vertices(1)
    val dz = translated.vertices(2) - centered.vertices(2)

    dx shouldBe offset.x +- 0.001f
    dy shouldBe offset.y +- 0.001f
    dz shouldBe offset.z +- 0.001f

  it should "scale geometry correctly" in:
    val small = TesseractMesh(size = 1.0f).toTriangleMesh
    val large = TesseractMesh(size = 2.0f).toTriangleMesh

    def boundingBoxSpan(mesh: menger.common.TriangleMeshData): Float =
      val xs = (0 until mesh.numVertices).map(i => mesh.vertices(i * 8))
      xs.max - xs.min

    val smallSpan = boundingBoxSpan(small)
    val largeSpan = boundingBoxSpan(large)

    // Projection causes non-linear scaling, so check that larger size produces larger span
    largeSpan should be > smallSpan

  it should "produce different geometry with XW rotation" in:
    val unrotated = TesseractMesh(rotXW = 0f, rotYW = 0f, rotZW = 0f).toTriangleMesh
    val rotated = TesseractMesh(rotXW = 45f, rotYW = 0f, rotZW = 0f).toTriangleMesh

    unrotated.vertices should not equal rotated.vertices

  it should "produce different geometry with YW rotation" in:
    val unrotated = TesseractMesh(rotXW = 0f, rotYW = 0f, rotZW = 0f).toTriangleMesh
    val rotated = TesseractMesh(rotXW = 0f, rotYW = 45f, rotZW = 0f).toTriangleMesh

    unrotated.vertices should not equal rotated.vertices

  it should "produce different geometry with ZW rotation" in:
    val unrotated = TesseractMesh(rotXW = 0f, rotYW = 0f, rotZW = 0f).toTriangleMesh
    val rotated = TesseractMesh(rotXW = 0f, rotYW = 0f, rotZW = 45f).toTriangleMesh

    unrotated.vertices should not equal rotated.vertices

  it should "produce same geometry for 0 and 360 degree rotation" in:
    val rot0 = TesseractMesh(rotXW = 0f, rotYW = 0f, rotZW = 0f).toTriangleMesh
    val rot360 = TesseractMesh(rotXW = 360f, rotYW = 0f, rotZW = 0f).toTriangleMesh

    rot0.vertices.zip(rot360.vertices).foreach { case (a, b) =>
      a shouldBe b +- 0.01f
    }

  it should "produce larger projection with closer eye" in:
    val far = TesseractMesh(eyeW = 10f, screenW = 5f).toTriangleMesh
    val near = TesseractMesh(eyeW = 3f, screenW = 1.5f).toTriangleMesh

    def maxExtent(mesh: menger.common.TriangleMeshData): Float =
      val xs = (0 until mesh.numVertices).map(i => mesh.vertices(i * 8).abs)
      xs.max

    maxExtent(near) should be > maxExtent(far)

  it should "handle edge-case projection distances" in:
    val farMesh = TesseractMesh(eyeW = 100f, screenW = 99f)
    noException should be thrownBy farMesh.toTriangleMesh

  it should "have visible 4D structure with default rotation" in:
    val mesh = TesseractMesh()
    mesh.rotXW shouldBe 15f
    mesh.rotYW shouldBe 10f
    val data = mesh.toTriangleMesh
    data.numTriangles shouldBe 48

  it should "reject invalid projection parameters (eyeW <= screenW)" in:
    an[IllegalArgumentException] should be thrownBy:
      TesseractMesh(eyeW = 1.0f, screenW = 2.0f)

  it should "reject eyeW equal to screenW" in:
    an[IllegalArgumentException] should be thrownBy:
      TesseractMesh(eyeW = 2.0f, screenW = 2.0f)

  it should "reject negative projection parameters" in:
    an[IllegalArgumentException] should be thrownBy:
      TesseractMesh(eyeW = -1.0f, screenW = 1.0f)

  it should "use identity rotation when all angles are zero" in:
    val mesh = TesseractMesh(rotXW = 0f, rotYW = 0f, rotZW = 0f)
    // Just verify it doesn't throw and produces valid output
    val data = mesh.toTriangleMesh
    data.numVertices shouldBe 96

  it should "have correct UV coordinates range" in:
    val mesh = TesseractMesh()
    val data = mesh.toTriangleMesh

    for i <- 0 until data.numVertices do
      val u = data.vertices(i * 8 + 6)
      val v = data.vertices(i * 8 + 7)
      u should (be >= 0f and be <= 1f)
      v should (be >= 0f and be <= 1f)
