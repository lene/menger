package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class Mesh4DProjectionSpec extends AnyFlatSpec with Matchers:

  // === Basic Construction Tests ===

  "Mesh4DProjection" should "accept a Tesseract mesh" in:
    val tesseract = Tesseract(size = 2.0f)
    val projection = Mesh4DProjection(
      mesh4D = tesseract,
      center = Vector3(0f, 0f, 0f),
      eyeW = 3.0f,
      screenW = 1.5f
    )
    val mesh = projection.toTriangleMesh
    mesh.numTriangles shouldBe 48
    mesh.numVertices shouldBe 96

  it should "work with default parameters" in:
    val tesseract = Tesseract(size = 1.0f)
    val projection = Mesh4DProjection(mesh4D = tesseract)
    val mesh = projection.toTriangleMesh
    mesh.numTriangles shouldBe 48

  it should "apply 4D rotation to mesh" in:
    val tesseract = Tesseract(size = 1.0f)
    val projection = Mesh4DProjection(
      mesh4D = tesseract,
      rotXW = 45f,
      rotYW = 30f,
      rotZW = 15f
    )
    val mesh = projection.toTriangleMesh
    mesh.numTriangles shouldBe 48
    mesh.vertices.length shouldBe 96 * 8

  it should "translate mesh to specified center" in:
    val tesseract = Tesseract(size = 1.0f)
    val center = Vector3(5f, -3f, 2f)
    val projection = Mesh4DProjection(
      mesh4D = tesseract,
      center = center,
      rotXW = 0f,
      rotYW = 0f,
      rotZW = 0f
    )
    val mesh = projection.toTriangleMesh

    // Check that vertices are offset (not all centered at origin)
    mesh.vertices.length shouldBe 96 * 8
    mesh.numTriangles shouldBe 48

  // === Validation Tests ===

  it should "require eyeW > screenW" in:
    val tesseract = Tesseract(size = 1.0f)
    an[IllegalArgumentException] should be thrownBy {
      Mesh4DProjection(
        mesh4D = tesseract,
        eyeW = 1.0f,
        screenW = 2.0f
      )
    }

  it should "reject equal eyeW and screenW" in:
    val tesseract = Tesseract(size = 1.0f)
    an[IllegalArgumentException] should be thrownBy {
      Mesh4DProjection(
        mesh4D = tesseract,
        eyeW = 2.0f,
        screenW = 2.0f
      )
    }

  it should "require positive eyeW and screenW" in:
    val tesseract = Tesseract(size = 1.0f)
    an[IllegalArgumentException] should be thrownBy {
      Mesh4DProjection(
        mesh4D = tesseract,
        eyeW = -1.0f,
        screenW = 0.5f
      )
    }

  // === Backward Compatibility Tests ===

  "TesseractMesh factory" should "create Mesh4DProjection with tesseract" in:
    val mesh = TesseractMesh(
      center = Vector3(0f, 0f, 0f),
      size = 2.0f,
      eyeW = 3.0f,
      screenW = 1.5f
    )
    mesh shouldBe a[Mesh4DProjection]
    val triangleMesh = mesh.toTriangleMesh
    triangleMesh.numTriangles shouldBe 48

  it should "work with default parameters" in:
    val mesh = TesseractMesh()
    mesh shouldBe a[Mesh4DProjection]
    val triangleMesh = mesh.toTriangleMesh
    triangleMesh.numTriangles shouldBe 48

  it should "apply rotation parameters" in:
    val mesh = TesseractMesh(
      size = 1.0f,
      rotXW = 45f,
      rotYW = 30f,
      rotZW = 15f
    )
    val triangleMesh = mesh.toTriangleMesh
    triangleMesh.numTriangles shouldBe 48

  it should "maintain same API as original TesseractMesh" in:
    val mesh = TesseractMesh(
      center = Vector3(1f, 2f, 3f),
      size = 2.5f,
      eyeW = 4.0f,
      screenW = 2.0f,
      rotXW = 30f,
      rotYW = 20f,
      rotZW = 10f
    )
    mesh.center shouldBe Vector3(1f, 2f, 3f)
    mesh.eyeW shouldBe 4.0f
    mesh.screenW shouldBe 2.0f
    mesh.rotXW shouldBe 30f
    mesh.rotYW shouldBe 20f
    mesh.rotZW shouldBe 10f

  // === Different 4D Mesh Types Tests ===

  "Mesh4DProjection with TesseractSponge" should "project volume-based sponge" in:
    val sponge = TesseractSponge(level = 1)
    val projection = Mesh4DProjection(mesh4D = sponge)
    val mesh = projection.toTriangleMesh

    // Level 1 TesseractSponge has more faces than basic tesseract
    mesh.numTriangles should be > 48
    mesh.numVertices should be > 96

  "Mesh4DProjection with TesseractSponge2" should "project surface-based sponge" in:
    val sponge = TesseractSponge2(level = 1, size = 1.0f)
    val projection = Mesh4DProjection(mesh4D = sponge)
    val mesh = projection.toTriangleMesh

    // Level 1 TesseractSponge2 has more faces than basic tesseract
    mesh.numTriangles should be > 48
    mesh.numVertices should be > 96

  // === Mesh Quality Tests ===

  "Generated mesh" should "have consistent vertex format" in:
    val tesseract = Tesseract(size = 1.0f)
    val projection = Mesh4DProjection(mesh4D = tesseract)
    val mesh = projection.toTriangleMesh

    // Vertex format: position(3) + normal(3) + uv(2) = 8 floats
    mesh.vertexStride shouldBe 8
    mesh.vertices.length shouldBe mesh.numVertices * 8

  it should "generate valid triangle indices" in:
    val tesseract = Tesseract(size = 1.0f)
    val projection = Mesh4DProjection(mesh4D = tesseract)
    val mesh = projection.toTriangleMesh

    // All indices should be valid (< numVertices)
    mesh.indices.foreach { index =>
      index should be >= 0
      index should be < mesh.numVertices
    }

  it should "have 3 indices per triangle" in:
    val tesseract = Tesseract(size = 1.0f)
    val projection = Mesh4DProjection(mesh4D = tesseract)
    val mesh = projection.toTriangleMesh

    mesh.indices.length shouldBe mesh.numTriangles * 3
