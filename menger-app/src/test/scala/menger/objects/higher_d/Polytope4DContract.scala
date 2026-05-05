package menger.objects.higher_d

import menger.common.Const
import org.scalatest.Inspectors.forAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Generic test contract for regular 4D polytopes.
  * Concrete suites extend AnyFlatSpec with Matchers with Polytope4DContract
  * and call registerPolytopeTests() at end of constructor. */
trait Polytope4DContract:
  this: AnyFlatSpec with Matchers =>

  val polytope: Mesh4D
  val polytopeLabel: String
  val expectedV: Int
  val expectedE: Int
  val expectedF: Int
  val expectedC: Int
  val expectedVertexNorm: Float
  val expectedEdgeLength: Float
  val expectedCellShape: Seq[menger.common.Vector[4]] => Unit

  protected def registerPolytopeTests(): Unit =
    val eps = Const.epsilon * 100f

    // A. Vertex
    polytopeLabel should s"have $expectedV vertices" in:
      polytope.vertices should have size expectedV

    it should "have no duplicate vertices" in:
      polytope.vertices.distinct should have size expectedV

    it should "have all vertices on a common 3-sphere" in:
      val norms = polytope.vertices.map(v => math.sqrt(v.len2).toFloat)
      forAll(norms) { _ should be(norms.head +- eps) }

    it should s"have vertex norm $expectedVertexNorm" in:
      forAll(polytope.vertices) { v =>
        math.sqrt(v.len2).toFloat should be(expectedVertexNorm +- eps)
      }

    it should "be centered at the origin" in:
      val centroid = polytope.vertices.reduce(_ + _) / polytope.vertices.size.toFloat
      forAll(centroid.toIndexedSeq) { _ should be(0f +- eps) }

    it should "have no NaN or Inf in vertex coordinates" in:
      forAll(polytope.vertices) { v =>
        forAll(v.toIndexedSeq) { c =>
          assert(!c.isNaN && !c.isInfinite, s"Invalid coord in $v")
        }
      }

    // B. Edge
    it should s"have $expectedE unique edges" in:
      polytope.edges should have size expectedE

    it should "have all edges of equal length" in:
      val lengths = polytope.edges.toSeq.map { case (a, b) => a.dst(b) }
      forAll(lengths) { _ should be(expectedEdgeLength +- eps) }

    it should "have all edge endpoints in the vertex set" in:
      val vset = polytope.vertices.toSet
      forAll(polytope.edges.toSeq) { case (a, b) =>
        vset should contain(a)
        vset should contain(b)
      }

    // C. Face
    it should s"have $expectedF faces" in:
      polytope.faces should have size expectedF

    it should "have no degenerate faces" in:
      forAll(polytope.faces) { f => f.area should be > 0f }

    it should "have all face vertices in the vertex set" in:
      val vset = polytope.vertices.toSet
      forAll(polytope.faces) { f =>
        forAll(f.asSeq) { v => vset should contain(v) }
      }

    it should "have all faces of equal area" in:
      val areas = polytope.faces.map(_.area)
      forAll(areas) { _ should be(areas.head +- eps) }

    // D. Cell
    it should s"have $expectedC cells" in:
      polytope.cells should have size expectedC

    it should "have cells of the correct shape" in:
      forAll(polytope.cells) { cell =>
        noException should be thrownBy expectedCellShape(cell)
      }

    // E. Topology
    it should "satisfy Euler-Poincaré: V - E + F - C = 0" in:
      val v = polytope.vertices.size
      val e = polytope.edges.size
      val f = polytope.faces.size
      val c = polytope.cells.size
      withClue(s"V=$v E=$e F=$f C=$c → ${v - e + f - c}") {
        (v - e + f - c) shouldBe 0
      }
