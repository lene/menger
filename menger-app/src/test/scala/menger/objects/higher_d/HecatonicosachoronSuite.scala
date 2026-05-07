package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class HecatonicosachoronSuite extends AnyFlatSpec with Matchers with Polytope4DContract:

  private val φ = ((1.0 + math.sqrt(5.0)) / 2.0).toFloat

  val polytope           = Hecatonicosachoron()
  val polytopeLabel      = "Hecatonicosachoron"
  val expectedV          = 600
  val expectedE          = 1200
  val expectedF          = 720
  val expectedC          = 120
  val expectedVertexNorm = 1.0f
  // centroid-dual edge length: sqrt((7-3*sqrt(5))/4) ≈ 0.270
  val expectedEdgeLength = math.sqrt((7.0 - 3.0 * math.sqrt(5.0)) / 4.0).toFloat
  val expectedCellShape  = CellShapeContract.assertDodecahedron

  registerPolytopeTests()

  it should "have exactly 120 dodecahedral cells with 20 vertices each" in:
    val cs = Hecatonicosachoron().cells
    cs should have size 120
    cs.foreach(_.size shouldBe 20)

  it should "have each pentagon edge shared by exactly 3 faces (closed 4D manifold)" in:
    val faces = Hecatonicosachoron().faces
    type VKey = (Float, Float, Float, Float)
    def vkey(v: menger.common.Vector[4]): VKey = (v(0), v(1), v(2), v(3))
    def ekey(a: VKey, b: VKey): (VKey, VKey) =
      if summon[Ordering[VKey]].lteq(a, b) then (a, b) else (b, a)

    val edgeCounts = faces.flatMap { f =>
      (0 until 5).map(i => ekey(vkey(f(i)), vkey(f((i + 1) % 5))))
    }.groupBy(identity).view.mapValues(_.size).toMap

    val counts = edgeCounts.values.toSet
    withClue(s"expected only edge-count 3, got ${counts}") {
      counts shouldBe Set(3)
    }
