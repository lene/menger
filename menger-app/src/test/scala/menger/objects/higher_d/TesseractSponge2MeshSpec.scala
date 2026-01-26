package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TesseractSponge2MeshSpec extends AnyFlatSpec with Matchers:

  // === Basic Construction Tests ===

  "TesseractSponge2Mesh" should "create level 0 mesh (basic tesseract)" in:
    val mesh = TesseractSponge2Mesh(level = 0)
    val triangleMesh = mesh.toTriangleMesh

    // Level 0 is a basic tesseract: 24 faces = 48 triangles
    triangleMesh.numTriangles shouldBe 48
    triangleMesh.numVertices shouldBe 96

  it should "create level 1 mesh" in:
    val mesh = TesseractSponge2Mesh(level = 1)
    val triangleMesh = mesh.toTriangleMesh

    // Level 1: 384 faces = 768 triangles
    triangleMesh.numTriangles shouldBe 768
    triangleMesh.numVertices shouldBe 1536

  it should "create level 2 mesh" in:
    val mesh = TesseractSponge2Mesh(level = 2)
    val triangleMesh = mesh.toTriangleMesh

    // Level 2: 6,144 faces = 12,288 triangles
    triangleMesh.numTriangles shouldBe 12288

  it should "create level 3 mesh" in:
    val mesh = TesseractSponge2Mesh(level = 3)
    val triangleMesh = mesh.toTriangleMesh

    // Level 3: 98,304 faces = 196,608 triangles
    triangleMesh.numTriangles shouldBe 196608

  it should "accept fractional levels" in:
    val mesh = TesseractSponge2Mesh(level = 1.5f)
    val triangleMesh = mesh.toTriangleMesh

    // Fractional level should truncate to level 1
    triangleMesh.numTriangles shouldBe 768

  it should "work with default parameters" in:
    val mesh = TesseractSponge2Mesh(level = 0)
    mesh shouldBe a[Mesh4DProjection]
    mesh.toTriangleMesh.numTriangles shouldBe 48

  it should "accept custom size parameter" in:
    val mesh = TesseractSponge2Mesh(size = 2.5f, level = 0)
    mesh shouldBe a[Mesh4DProjection]

  it should "accept custom 4D projection parameters" in:
    val mesh = TesseractSponge2Mesh(
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
    val mesh = TesseractSponge2Mesh(
      center = center,
      level = 0
    )
    mesh.center shouldBe center

  // === Face Count Estimation Tests ===

  "TesseractSponge2Mesh.estimatedFaces" should "calculate level 0 count" in:
    TesseractSponge2Mesh.estimatedFaces(0) shouldBe 24L

  it should "calculate level 1 count" in:
    TesseractSponge2Mesh.estimatedFaces(1) shouldBe 384L

  it should "calculate level 2 count" in:
    TesseractSponge2Mesh.estimatedFaces(2) shouldBe 6144L

  it should "calculate level 3 count" in:
    TesseractSponge2Mesh.estimatedFaces(3) shouldBe 98304L

  it should "calculate level 4 count" in:
    // 24 * 16^4 = 1,572,864
    TesseractSponge2Mesh.estimatedFaces(4) shouldBe 1572864L

  it should "calculate level 5 count" in:
    // 24 * 16^5 = 25,165,824
    TesseractSponge2Mesh.estimatedFaces(5) shouldBe 25165824L

  // === Triangle Count Estimation Tests ===

  "TesseractSponge2Mesh.estimatedTriangles" should "be double the face count" in:
    TesseractSponge2Mesh.estimatedTriangles(0) shouldBe 48L
    TesseractSponge2Mesh.estimatedTriangles(1) shouldBe 768L
    TesseractSponge2Mesh.estimatedTriangles(2) shouldBe 12288L

  it should "match actual mesh triangle count at level 0" in:
    val mesh = TesseractSponge2Mesh(level = 0)
    val triangleMesh = mesh.toTriangleMesh
    TesseractSponge2Mesh.estimatedTriangles(0) shouldBe triangleMesh.numTriangles

  it should "match actual mesh triangle count at level 1" in:
    val mesh = TesseractSponge2Mesh(level = 1)
    val triangleMesh = mesh.toTriangleMesh
    TesseractSponge2Mesh.estimatedTriangles(1) shouldBe triangleMesh.numTriangles

  // === Mesh Quality Tests ===

  "Generated mesh" should "have consistent vertex format" in:
    val mesh = TesseractSponge2Mesh(level = 1)
    val triangleMesh = mesh.toTriangleMesh

    // Vertex format: position(3) + normal(3) + uv(2) = 8 floats
    triangleMesh.vertexStride shouldBe 8
    triangleMesh.vertices.length shouldBe triangleMesh.numVertices * 8

  it should "generate valid triangle indices" in:
    val mesh = TesseractSponge2Mesh(level = 1)
    val triangleMesh = mesh.toTriangleMesh

    // All indices should be valid (< numVertices)
    triangleMesh.indices.foreach { index =>
      index should be >= 0
      index should be < triangleMesh.numVertices
    }

  it should "have 3 indices per triangle" in:
    val mesh = TesseractSponge2Mesh(level = 1)
    val triangleMesh = mesh.toTriangleMesh

    triangleMesh.indices.length shouldBe triangleMesh.numTriangles * 3

  // === Growth Pattern Tests ===

  "Level growth" should "follow 16x pattern" in:
    val level0Faces = TesseractSponge2Mesh.estimatedFaces(0)
    val level1Faces = TesseractSponge2Mesh.estimatedFaces(1)
    val level2Faces = TesseractSponge2Mesh.estimatedFaces(2)

    level1Faces shouldBe level0Faces * 16
    level2Faces shouldBe level1Faces * 16

  it should "produce more faces at each level" in:
    val mesh0 = TesseractSponge2Mesh(level = 0).toTriangleMesh
    val mesh1 = TesseractSponge2Mesh(level = 1).toTriangleMesh
    val mesh2 = TesseractSponge2Mesh(level = 2).toTriangleMesh

    mesh1.numTriangles should be > mesh0.numTriangles
    mesh2.numTriangles should be > mesh1.numTriangles

  // === Comparison with Tesseract Tests ===

  "TesseractSponge2Mesh level 0" should "match basic tesseract" in:
    val spongeMesh = TesseractSponge2Mesh(level = 0).toTriangleMesh
    val tesseractMesh = TesseractMesh().toTriangleMesh

    spongeMesh.numTriangles shouldBe tesseractMesh.numTriangles
    spongeMesh.numVertices shouldBe tesseractMesh.numVertices

  // === Comparison with TesseractSponge Tests ===

  "TesseractSponge2" should "have fewer faces than TesseractSponge at same level" in:
    // TesseractSponge grows as 48^level (volume-based)
    // TesseractSponge2 grows as 16^level (surface-based)
    // So TesseractSponge2 should have fewer faces at higher levels

    val sponge1 = TesseractSpongeMesh.estimatedFaces(2)
    val sponge2 = TesseractSponge2Mesh.estimatedFaces(2)

    sponge2 should be < sponge1
