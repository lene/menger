package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class HexacosichoronSuite extends AnyFlatSpec with Matchers with Polytope4DContract:

  private val φ = ((1.0 + math.sqrt(5.0)) / 2.0).toFloat

  val polytope           = Hexacosichoron()
  val polytopeLabel      = "Hexacosichoron"
  val expectedV          = 120
  val expectedE          = 720
  val expectedF          = 1200
  val expectedC          = 600
  val expectedVertexNorm = 1.0f              // circumradius = size = 1
  val expectedEdgeLength = (1f / φ)         // edge = size/φ ≈ 0.618
  val expectedCellShape  = CellShapeContract.assertTetrahedron

  registerPolytopeTests()

  it should "have exactly 120 distinct vertices" in:
    Hexacosichoron().vertices.distinct should have size 120

  it should "include 8 axis, 16 hypercube, and 96 golden vertices" in:
    val verts = Hexacosichoron().vertices
    val norms = verts.map(v => (0 until 4).map(i => v(i) * v(i)).sum)
    norms.foreach(_ should be(1f +- 0.001f))

  it should "have no boundary edges (closed 4D polytope)" in:
    val tris = Hexacosichoron().faces.map(f => Face4D[3](IndexedSeq(f(0), f(1), f(2))))
    val report = MeshTopology.checkTriangle4D(tris)
    withClue(s"boundary edge count: ${report.boundaryEdgeCount}") {
      report.boundaryEdgeCount shouldBe 0
    }
