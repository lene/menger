package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class HexadecachoronSuite extends AnyFlatSpec with Matchers with Polytope4DContract:

  val polytope            = Hexadecachoron()
  val polytopeLabel       = "Hexadecachoron"
  val expectedV           = 8
  val expectedE           = 24
  val expectedF           = 32
  val expectedC           = 16
  val expectedVertexNorm  = (1.0 / math.sqrt(2)).toFloat   // s = 1/√2 for size=1
  val expectedEdgeLength  = 1.0f                            // s√2 = 1
  val expectedCellShape   = CellShapeContract.assertTetrahedron

  registerPolytopeTests()

  it should "have 8 vertices as signed unit-axis vectors" in:
    val vset = Hexadecachoron().vertices.map(v => (v(0), v(1), v(2), v(3))).toSet
    val s = expectedVertexNorm
    Set((s,0f,0f,0f), (-s,0f,0f,0f),
        (0f,s,0f,0f), (0f,-s,0f,0f),
        (0f,0f,s,0f), (0f,0f,-s,0f),
        (0f,0f,0f,s), (0f,0f,0f,-s)) shouldBe vset

  it should "not connect antipodal vertex pairs" in:
    val vs = Hexadecachoron().vertices
    val antipodePairs = vs.zip(vs.map(v => -v))
    val edgeSet = Hexadecachoron().edges
    antipodePairs.foreach { case (a, b) =>
      val ka = (a(0),a(1),a(2),a(3))
      val kb = (b(0),b(1),b(2),b(3))
      val canon = if summon[Ordering[(Float,Float,Float,Float)]].lteq(ka,kb) then (a,b) else (b,a)
      edgeSet should not contain canon
    }

  it should "have no boundary edges (closed 4D polytope)" in:
    val tris = Hexadecachoron().faces.map(f => Face4D[3](IndexedSeq(f(0), f(1), f(2))))
    val report = MeshTopology.checkTriangle4D(tris)
    withClue(s"boundary edges: ${report.boundaryEdgeCount}, histogram: ${report.edgeUseHistogram}") {
      report.boundaryEdgeCount shouldBe 0
    }
