package menger.objects.higher_d

import menger.common.Vector
import org.scalatest.Inspectors.forAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class IcositetrachoronSuite extends AnyFlatSpec with Matchers with Polytope4DContract:

  val polytope            = Icositetrachoron()
  val polytopeLabel       = "Icositetrachoron"
  val expectedV           = 24
  val expectedE           = 96
  val expectedF           = 96
  val expectedC           = 24
  val expectedVertexNorm  = 0.5f   // s = size/2 = 0.5 for size=1
  val expectedEdgeLength  = 0.5f   // axial-to-half and half-to-adjacent-half = 0.5
  val expectedCellShape   = CellShapeContract.assertOctahedron

  registerPolytopeTests()

  it should "have 8 axial vertices and 16 half vertices" in:
    val vs     = Icositetrachoron().vertices
    val s      = 0.5f
    val axial  = vs.filter(v => v.toIndexedSeq.count(_ != 0f) == 1)
    val half   = vs.filter(v => v.toIndexedSeq.forall(_ != 0f))
    axial should have size 8
    half  should have size 16

  it should "have all axial vertex coordinates ±0.5 or 0" in:
    val axial = Icositetrachoron().vertices.filter(v => v.toIndexedSeq.count(_ != 0f) == 1)
    forAll(axial) { v =>
      val nonzero = v.toIndexedSeq.filter(_ != 0f)
      nonzero should have size 1
      nonzero.head.abs should be(0.5f +- 1e-4f)
    }

  it should "pass manifold check on triangular faces after fix" in:
    val report = MeshTopology.checkTriangle4D(
      Icositetrachoron().faces.asInstanceOf[Seq[Face4D[3]]])
    withClue(s"boundary edges: ${report.boundaryEdgeCount}, histogram: ${report.edgeUseHistogram}") {
      report.isManifold shouldBe true
    }
