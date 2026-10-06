package menger.objects

import menger.common.ProfilingConfig
import menger.common.TriangleMeshData
import menger.common.Vector
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Usability review 2026-09, session 2 (F35): a fractional level used to lay the whole level-n
  * surface, pushed 0.0003 outward, over the level-(n+1) surface at alpha 1 - frac. The two
  * near-coincident surfaces gave speckle, double refraction on glass and shifted procedural
  * colours (F41). Now only the hole caps fade, and no transparent face overlaps an opaque one. */
class FractionalHoleCapsSuite extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig.disabled

  private val PlaneTolerance = 1e-3f
  private val AreaTolerance = 1e-5f

  private case class Quad(min: Seq[Float], max: Seq[Float], normalAxis: Int, alpha: Float):
    def extent(axis: Int): Float = max(axis) - min(axis)

  private def quads(mesh: TriangleMeshData): Seq[Quad] =
    val stride = mesh.vertexStride
    (0 until mesh.numVertices / 4).map { q =>
      val bases = (0 until 4).map(i => (q * 4 + i) * stride)
      def coords(axis: Int) = bases.map(b => mesh.vertices(b + axis))
      val normalAxis = (0 until 3).maxBy(a => math.abs(mesh.vertices(bases.head + 3 + a)))
      Quad((0 until 3).map(coords(_).min), (0 until 3).map(coords(_).max), normalAxis,
        mesh.vertices(bases.head + 8))
    }

  private def overlaps(t: Quad, o: Quad): Boolean =
    val axis = t.normalAxis
    t.normalAxis == o.normalAxis && math.abs(t.min(axis) - o.min(axis)) < PlaneTolerance &&
      (0 until 3).filter(_ != axis).forall { a =>
        math.min(t.max(a), o.max(a)) - math.max(t.min(a), o.min(a)) > AreaTolerance
      }

  private def transparentAndOpaque(mesh: TriangleMeshData): (Seq[Quad], Seq[Quad]) =
    quads(mesh).partition(_.alpha < 1f)

  private def surface(level: Float) = SpongeBySurface(Vector.Zero[3], 1f, level)
  private def volume(level: Float) = SpongeByVolume(Vector.Zero[3], 1f, level)

  Seq(0.5f, 1.5f).foreach { level =>
    s"A fractional SpongeBySurface at level $level" should
      "fade no face that lies over an opaque one" in:
        val (transparent, opaque) = transparentAndOpaque(surface(level).toTriangleMesh)
        transparent should not be empty
        transparent.count(t => opaque.exists(overlaps(t, _))) shouldBe 0

    s"A fractional SpongeByVolume at level $level" should
      "fade no face that lies over an opaque one" in:
        val (transparent, opaque) = transparentAndOpaque(volume(level).toTriangleMesh)
        transparent should not be empty
        transparent.count(t => opaque.exists(overlaps(t, _))) shouldBe 0
  }

  "The hole caps of SpongeBySurface level 0.5" should "be the centre third of each cube face" in:
    val (transparent, _) = transparentAndOpaque(surface(0.5f).toTriangleMesh)
    transparent should have size 6
    all(transparent.map(_.alpha)) shouldBe 0.5f
    transparent.foreach { cap =>
      (0 until 3).filter(_ != cap.normalAxis).foreach { a =>
        cap.extent(a) shouldBe (1f / 3f +- 1e-5f)
        (cap.min(a) + cap.max(a)) shouldBe (0f +- 1e-5f)
      }
    }

  they should "count one per exposed level-n face" in:
    val capCount = transparentAndOpaque(volume(1.25f).toTriangleMesh)._1.size
    capCount shouldBe volume(1f).toTriangleMesh.numVertices / 4
