package menger.objects.higher_d

import menger.common.Vector
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PentachoronSuite extends AnyFlatSpec with Matchers with Polytope4DContract:

  val polytope            = Pentachoron()
  val polytopeLabel       = "Pentachoron"
  val expectedV           = 5
  val expectedE           = 10
  val expectedF           = 10
  val expectedC           = 5
  val expectedVertexNorm  = math.sqrt(0.8).toFloat   // 2/√5 · size/2 · 2 = √0.8
  val expectedEdgeLength  = math.sqrt(2.0).toFloat   // √2 for size=1
  val expectedCellShape   = CellShapeContract.assertTetrahedron

  registerPolytopeTests()

  it should "scale vertex norm linearly with size" in:
    val scale = 2f
    val base  = math.sqrt(Pentachoron().vertices.map(_.len2).sum / 5).toFloat
    val scaled = math.sqrt(Pentachoron(scale).vertices.map(_.len2).sum / 5).toFloat
    scaled should be(base * scale +- 1e-4f)

  it should "have apex on +w axis (w > 0, xyz = 0)" in:
    val apex = Pentachoron().vertices.maxBy(v => v(3))
    apex(0) should be(0f +- 1e-4f)
    apex(1) should be(0f +- 1e-4f)
    apex(2) should be(0f +- 1e-4f)
    apex(3) should be > 0f

  it should "pass manifold check on triangular faces after fix" in:
    val report = MeshTopology.checkTriangle4D(
      Pentachoron().faces.asInstanceOf[Seq[Face4D[3]]])
    withClue(s"boundary edges: ${report.boundaryEdgeCount}, histogram: ${report.edgeUseHistogram}") {
      report.isManifold shouldBe true
    }
