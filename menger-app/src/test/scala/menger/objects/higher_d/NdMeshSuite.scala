package menger.objects.higher_d

import menger.common.Vector
import menger.objects.higher_d.CustomMatchers.closeTo
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NdMeshSuite extends AnyFlatSpec with Matchers:

  private val p0 = Vector[5](0f, 0f, 0f, 0f, 1f)
  private val p1 = Vector[5](1f, 0f, 0f, 0f, 0f)
  private val p2 = Vector[5](0f, 1f, 0f, 0f, 0f)
  private val p3 = Vector[5](0f, 0f, 0f, 1f, 0f)

  private val tetrahedron = new Mesh[5]:
    type V = 3
    val vertices: Seq[Vector[5]] = Seq(p0, p1, p2, p3)
    lazy val faces: Seq[Polygon[5, V]] = Seq(
      Polygon[5, 3](IndexedSeq(p0, p1, p2)), Polygon[5, 3](IndexedSeq(p0, p1, p3)),
      Polygon[5, 3](IndexedSeq(p0, p2, p3)), Polygon[5, 3](IndexedSeq(p1, p2, p3))
    )

  "A Mesh[5]" should "list every edge of its faces once" in:
    tetrahedron.edges.size shouldBe 6

  it should "order each edge's endpoints lexicographically by coordinate" in:
    tetrahedron.edges.foreach { (from, to) =>
      from.toIndexedSeq.zip(to.toIndexedSeq).find(_ != _).forall(_ < _) shouldBe true
    }

  "A Polygon[5, 3]" should "reject a wrong vertex count, naming the dimension" in:
    val wrong = the[IllegalArgumentException] thrownBy Polygon[5, 3](IndexedSeq(p0, p1))
    wrong.getMessage should (include("Polygon[5, 3]") and include("got 2"))

  it should "have the area of the parallelogram its first two edges span" in:
    Polygon[5, 3](IndexedSeq(p0, p1, p2)).area shouldBe math.sqrt(3.0).toFloat +- 1e-5f

  "A Rotation[5]" should "rotate every vertex of a Polygon[5, V]" in:
    given ValueOf[5] = ValueOf(5)
    val polygon = Polygon[5, 3](IndexedSeq(p0, p1, p2))
    val rotation = Rotation.inPlane[5](Plane(0, 4, 5), 90f)
    val rotated = rotation(polygon)
    rotated.vertices.zip(polygon.vertices).foreach { (r, p) => r should closeTo(rotation(p)) }
    rotated(1) should closeTo(Vector[5](0f, 0f, 0f, 0f, -1f))
