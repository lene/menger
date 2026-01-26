package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TesseractSpongeMeshSpec extends AnyFlatSpec with Matchers:

  // === Basic Construction Tests ===

  "TesseractSpongeMesh" should "create level 0 mesh (basic tesseract)" in:
    val mesh = TesseractSpongeMesh(level = 0)
    val triangleMesh = mesh.toTriangleMesh

    // Level 0 is a basic tesseract: 24 faces = 48 triangles
    triangleMesh.numTriangles shouldBe 48
    triangleMesh.numVertices shouldBe 96

  it should "create level 1 mesh" in:
    val mesh = TesseractSpongeMesh(level = 1)
    val triangleMesh = mesh.toTriangleMesh

    // Level 1: 1,152 faces = 2,304 triangles
    triangleMesh.numTriangles shouldBe 2304
    triangleMesh.numVertices shouldBe 4608

  it should "create level 2 mesh" in:
    val mesh = TesseractSpongeMesh(level = 2)
    val triangleMesh = mesh.toTriangleMesh

    // Level 2: 55,296 faces = 110,592 triangles
    triangleMesh.numTriangles shouldBe 110592

  it should "accept fractional levels" in:
    val mesh = TesseractSpongeMesh(level = 1.5f)
    val triangleMesh = mesh.toTriangleMesh

    // Fractional level should truncate to level 1
    triangleMesh.numTriangles shouldBe 2304

  it should "work with default parameters" in:
    val mesh = TesseractSpongeMesh(level = 0)
    mesh shouldBe a[Mesh4DProjection]
    mesh.toTriangleMesh.numTriangles shouldBe 48

  it should "accept custom 4D projection parameters" in:
    val mesh = TesseractSpongeMesh(
      level = 1,
      eyeW = 5.0f,
      screenW = 2.5f,
      rotXW = 45f,
      rotYW = 30f,
      rotZW = 15f
    )
    mesh.eyeW shouldBe 5.0f
    mesh.screenW shouldBe 2.5f
    mesh.rotXW shouldBe 45f
    mesh.rotYW shouldBe 30f
    mesh.rotZW shouldBe 15f

  it should "translate mesh to specified center" in:
    val center = Vector3(5f, -3f, 2f)
    val mesh = TesseractSpongeMesh(
      center = center,
      level = 0
    )
    mesh.center shouldBe center

  // === Face Count Estimation Tests ===

  "TesseractSpongeMesh.estimatedFaces" should "calculate level 0 count" in:
    TesseractSpongeMesh.estimatedFaces(0) shouldBe 24L

  it should "calculate level 1 count" in:
    TesseractSpongeMesh.estimatedFaces(1) shouldBe 1152L

  it should "calculate level 2 count" in:
    TesseractSpongeMesh.estimatedFaces(2) shouldBe 55296L

  it should "calculate level 3 count" in:
    TesseractSpongeMesh.estimatedFaces(3) shouldBe 2654208L

  it should "calculate level 4 count" in:
    // 24 * 48^4 = 127,401,984
    TesseractSpongeMesh.estimatedFaces(4) shouldBe 127401984L

  // === Triangle Count Estimation Tests ===

  "TesseractSpongeMesh.estimatedTriangles" should "be double the face count" in:
    TesseractSpongeMesh.estimatedTriangles(0) shouldBe 48L
    TesseractSpongeMesh.estimatedTriangles(1) shouldBe 2304L
    TesseractSpongeMesh.estimatedTriangles(2) shouldBe 110592L

  it should "match actual mesh triangle count at level 0" in:
    val mesh = TesseractSpongeMesh(level = 0)
    val triangleMesh = mesh.toTriangleMesh
    TesseractSpongeMesh.estimatedTriangles(0) shouldBe triangleMesh.numTriangles

  it should "match actual mesh triangle count at level 1" in:
    val mesh = TesseractSpongeMesh(level = 1)
    val triangleMesh = mesh.toTriangleMesh
    TesseractSpongeMesh.estimatedTriangles(1) shouldBe triangleMesh.numTriangles

  // === Mesh Quality Tests ===

  "Generated mesh" should "have consistent vertex format" in:
    val mesh = TesseractSpongeMesh(level = 1)
    val triangleMesh = mesh.toTriangleMesh

    // Vertex format: position(3) + normal(3) + uv(2) = 8 floats
    triangleMesh.vertexStride shouldBe 8
    triangleMesh.vertices.length shouldBe triangleMesh.numVertices * 8

  it should "generate valid triangle indices" in:
    val mesh = TesseractSpongeMesh(level = 1)
    val triangleMesh = mesh.toTriangleMesh

    // All indices should be valid (< numVertices)
    triangleMesh.indices.foreach { index =>
      index should be >= 0
      index should be < triangleMesh.numVertices
    }

  it should "have 3 indices per triangle" in:
    val mesh = TesseractSpongeMesh(level = 1)
    val triangleMesh = mesh.toTriangleMesh

    triangleMesh.indices.length shouldBe triangleMesh.numTriangles * 3

  // === Growth Pattern Tests ===

  "Level growth" should "follow 48x pattern" in:
    val level0Faces = TesseractSpongeMesh.estimatedFaces(0)
    val level1Faces = TesseractSpongeMesh.estimatedFaces(1)
    val level2Faces = TesseractSpongeMesh.estimatedFaces(2)

    level1Faces shouldBe level0Faces * 48
    level2Faces shouldBe level1Faces * 48

  it should "produce more faces at each level" in:
    val mesh0 = TesseractSpongeMesh(level = 0).toTriangleMesh
    val mesh1 = TesseractSpongeMesh(level = 1).toTriangleMesh

    mesh1.numTriangles should be > mesh0.numTriangles

  // === Comparison with Tesseract Tests ===

  "TesseractSpongeMesh level 0" should "match basic tesseract" in:
    val spongeMesh = TesseractSpongeMesh(level = 0).toTriangleMesh
    val tesseractMesh = TesseractMesh().toTriangleMesh

    spongeMesh.numTriangles shouldBe tesseractMesh.numTriangles
    spongeMesh.numVertices shouldBe tesseractMesh.numVertices
