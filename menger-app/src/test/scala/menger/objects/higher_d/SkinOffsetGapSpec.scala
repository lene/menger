package menger.objects.higher_d

import scala.collection.mutable

import menger.common.Vector
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Regression detector for the fractional-4D-sponge corner artifact
  * (investigation 2026-06-03).
  *
  * The fractional skin offset must not split vertices that are coincident in the
  * source mesh: if two faces sharing a vertex move it to different places, the
  * gap shows the level-(n+1) surface behind as an extra dark square at the cube
  * corners. The original `quadsBuffer` offset (`Σ winding-signed face normals`)
  * did exactly that; the radial-scale offset is gap-free by construction.
  *
  * `countGaps` reconstructs the offset positions from the real `quadsBuffer`
  * output and counts source-coincident vertex groups that diverge. */
@SuppressWarnings(Array("org.wartremover.warts.AsInstanceOf"))
class SkinOffsetGapSpec extends AnyFlatSpec with Matchers:

  private val Eps = 1e-3f

  private def key(v: Vector[4]): (Int, Int, Int, Int) =
    def r(x: Float): Int = math.round(x * 1000f)
    (r(v(0)), r(v(1)), r(v(2)), r(v(3)))

  /** Count source-coincident vertices that land in different places in `buffer`.
    * `buffer` is the flat `quadsBuffer` output: 4 floats per vertex, 4 vertices
    * per face, in face order. */
  private def countGaps(faces: Seq[Face4D[4]], buffer: Array[Float]): Int =
    val groups = mutable.Map[(Int, Int, Int, Int), mutable.ListBuffer[Vector[4]]]()
    faces.iterator.zipWithIndex.foreach { case (f, fi) =>
      (0 until 4).foreach { j =>
        val base = (fi * 4 + j) * 4
        val moved = Vector[4](buffer(base), buffer(base + 1), buffer(base + 2), buffer(base + 3))
        groups.getOrElseUpdate(key(f(j)), mutable.ListBuffer.empty) += moved
      }
    }
    groups.count { case (_, moved) =>
      moved.sizeIs > 1 && moved.combinations(2).exists { p => (p(0) - p(1)).len > Eps }
    }

  private def spongeFaces(level: Int): Seq[Face4D[4]] =
    new TesseractSponge(level).faces.map(_.asInstanceOf[Face4D[4]])

  private def spongeGaps(level: Int, offset: Float): Int =
    countGaps(spongeFaces(level), Mesh4DGpuFlatten.quadsBuffer(new TesseractSponge(level), offset))

  // === detector self-validation ===

  private val sharedFace = Face4D[4](IndexedSeq(
    Vector[4](0f, 0f, 0f, 0f), Vector[4](1f, 0f, 0f, 0f),
    Vector[4](1f, 1f, 0f, 0f), Vector[4](0f, 1f, 0f, 0f)))

  "countGaps" should "report 0 when a shared vertex lands in the same place" in:
    // two faces, both with v0=(0,0,0,0); buffer maps both v0 copies to (0,0,0,5)
    val buffer = Array.fill(32)(0f)
    buffer(2) = 5f   // face0 v0 -> (0,0,5,0)
    buffer(18) = 5f  // face1 v0 -> (0,0,5,0)  (same place: no gap)
    countGaps(Seq(sharedFace, sharedFace), buffer) shouldBe 0

  it should "report a gap when a shared vertex lands in two places" in:
    val buffer = Array.fill(32)(0f)
    buffer(2) = 5f   // face0 v0 -> (0,0,5,0)
    buffer(18) = 9f  // face1 v0 -> (0,0,9,0)  (different: gap)
    countGaps(Seq(sharedFace, sharedFace), buffer) should be > 0

  // === regression: real quadsBuffer must be gap-free at every level ===

  "quadsBuffer skin offset" should "open no gaps at sponge levels 0, 1, 2" in:
    Seq(0, 1, 2).foreach { lvl =>
      withClue(s"level $lvl: ") { spongeGaps(lvl, 0.05f) shouldBe 0 }
    }
