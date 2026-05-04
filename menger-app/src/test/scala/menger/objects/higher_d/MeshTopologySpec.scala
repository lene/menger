package menger.objects.higher_d

import menger.common.{TriangleMeshData, Vector}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MeshTopologySpec extends AnyFlatSpec with Matchers:

  // -------------------------------------------------------------------------
  // Fixtures
  // -------------------------------------------------------------------------

  // 3D unit cube: 8 vertices, 12 triangles (2 per face), stride=6 (pos+normal).
  // Normals are all zeros — the checker only uses positions.
  private val cubeVertices: Array[Float] =
    Array(
      0f,0f,0f, 0f,0f,0f,  // v0
      1f,0f,0f, 0f,0f,0f,  // v1
      1f,1f,0f, 0f,0f,0f,  // v2
      0f,1f,0f, 0f,0f,0f,  // v3
      0f,0f,1f, 0f,0f,0f,  // v4
      1f,0f,1f, 0f,0f,0f,  // v5
      1f,1f,1f, 0f,0f,0f,  // v6
      0f,1f,1f, 0f,0f,0f   // v7
    )

  // Two triangles per face, covering all 6 faces outward-consistently.
  private val cubeIndices: Array[Int] =
    Array(
      0,1,2,  0,2,3,  // front  z=0
      5,4,7,  5,7,6,  // back   z=1
      0,4,5,  0,5,1,  // bottom y=0
      3,2,6,  3,6,7,  // top    y=1
      0,3,7,  0,7,4,  // left   x=0
      1,5,6,  1,6,2   // right  x=1
    )

  private val unitCube = TriangleMeshData(cubeVertices, cubeIndices, vertexStride = 6)

  // Same cube with triangle 0 (0,1,2) removed — 3 edges become boundary.
  private val brokenCube = TriangleMeshData(
    cubeVertices,
    cubeIndices.drop(3),  // remove first triangle's 3 indices
    vertexStride = 6
  )

  // 4D cube surface: 6 axis-aligned quads forming a closed 2-manifold.
  // Vertices are the 3D unit-cube corners padded to 4D with w=0.
  // Face4D(a,b,c,d) — direct 4-arg constructor, no validation triggered.
  private def v(x: Float, y: Float, z: Float): Vector[4] = Vector[4](x, y, z, 0f)
  private val p = Array(
    v(0,0,0), v(1,0,0), v(1,1,0), v(0,1,0),  // 0-3: z=0 corners
    v(0,0,1), v(1,0,1), v(1,1,1), v(0,1,1)   // 4-7: z=1 corners
  )
  private val cube4DFaces: Seq[Face4D[4]] = Seq(
    Face4D(p(0), p(1), p(2), p(3)),  // 0: bottom z=0
    Face4D(p(4), p(5), p(6), p(7)),  // 1: top    z=1
    Face4D(p(0), p(1), p(5), p(4)),  // 2: front  y=0
    Face4D(p(3), p(2), p(6), p(7)),  // 3: back   y=1
    Face4D(p(0), p(3), p(7), p(4)),  // 4: left   x=0
    Face4D(p(1), p(2), p(6), p(5))   // 5: right  x=1
  )
  // Each of the 12 edges is shared by exactly 2 faces → isManifold.

  private val brokenCube4D: Seq[Face4D[4]] = cube4DFaces.tail  // drop face 0

  // -------------------------------------------------------------------------
  // checkTriangleMesh — known-good (unit cube)
  // -------------------------------------------------------------------------

  "checkTriangleMesh" should "report a unit cube as manifold" in:
    val report = MeshTopology.checkTriangleMesh(unitCube)
    report.faceCount shouldBe 12
    report.isManifold shouldBe true
    report.boundaryFaces shouldBe empty

  it should "report edge-use histogram Map(2->18) for the unit cube" in:
    val report = MeshTopology.checkTriangleMesh(unitCube)
    report.edgeUseHistogram shouldBe Map(2 -> 18)

  // -------------------------------------------------------------------------
  // checkTriangleMesh — known-bad (cube minus one triangle)
  // -------------------------------------------------------------------------

  it should "report a unit cube with one triangle removed as non-manifold" in:
    val report = MeshTopology.checkTriangleMesh(brokenCube)
    report.faceCount shouldBe 11
    report.isManifold shouldBe false

  it should "report exactly 3 boundary edges when one triangle is removed" in:
    val report = MeshTopology.checkTriangleMesh(brokenCube)
    report.boundaryEdgeCount shouldBe 3

  it should "identify the 3 triangles that share those boundary edges" in:
    val report = MeshTopology.checkTriangleMesh(brokenCube)
    // Removed triangle was (v0,v1,v2). Its three edges were shared with:
    //   {v0,v1} → triangle (v0,v5,v1)  = new index 4
    //   {v1,v2} → triangle (v1,v6,v2)  = new index 10
    //   {v0,v2} → triangle (v0,v2,v3)  = new index 0
    report.boundaryFaces shouldBe Seq(0, 4, 10)

  it should "report histogram Map(1->3, 2->15) for the broken cube" in:
    val report = MeshTopology.checkTriangleMesh(brokenCube)
    report.edgeUseHistogram shouldBe Map(1 -> 3, 2 -> 15)

  // -------------------------------------------------------------------------
  // checkFace4D — known-good (6-quad cube surface in 4D)
  // -------------------------------------------------------------------------

  "checkFace4D" should "report a 6-quad cube surface as manifold" in:
    val report = MeshTopology.checkFace4D(cube4DFaces)
    report.faceCount shouldBe 6
    report.isManifold shouldBe true
    report.boundaryFaces shouldBe empty

  it should "report edge-use histogram Map(2->12) for the 6-quad cube surface" in:
    val report = MeshTopology.checkFace4D(cube4DFaces)
    report.edgeUseHistogram shouldBe Map(2 -> 12)

  // -------------------------------------------------------------------------
  // checkFace4D — known-bad (cube surface minus one face)
  // -------------------------------------------------------------------------

  it should "report a 5-face surface (one face removed) as non-manifold" in:
    val report = MeshTopology.checkFace4D(brokenCube4D)
    report.faceCount shouldBe 5
    report.isManifold shouldBe false

  it should "report exactly 4 boundary edges when one quad is removed" in:
    val report = MeshTopology.checkFace4D(brokenCube4D)
    report.boundaryEdgeCount shouldBe 4

  it should "identify the 4 remaining faces that share the boundary edges" in:
    val report = MeshTopology.checkFace4D(brokenCube4D)
    // Removed face 0 (bottom) shared one edge with each of face 1..4 (indices
    // 0..3 in the remaining 5-face list after dropping face 0, i.e. tail).
    report.boundaryFaces shouldBe Seq(1, 2, 3, 4)
