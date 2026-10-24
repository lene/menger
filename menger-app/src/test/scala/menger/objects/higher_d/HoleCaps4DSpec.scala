package menger.objects.higher_d

import menger.common.Vector
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Usability review 2026-09, session 2 (F35): a fractional 4D sponge used to upload the whole
  * level-n mesh, radially scaled by 1.0003, over the level-(n+1) mesh at alpha 1 - frac. Now it
  * uploads only the hole caps, the centre third of each level-n face, which the level-(n+1)
  * sponge leaves open, so no transparent face lies over an opaque one. Replaces
  * `SkinOffsetGapSpec`, which guarded the removed radial skin offset. */
class HoleCaps4DSpec extends AnyFlatSpec with Matchers:

  private val Tolerance = 1e-4f
  private val FloatsPerFace = 16

  private type Quad = Seq[Vector[4]]

  private def capQuads(mesh: Mesh4D): Seq[Quad] =
    Mesh4DGpuFlatten.holeCapsBuffer(mesh).grouped(FloatsPerFace).map { face =>
      face.grouped(4).map(v => Vector[4](v(0), v(1), v(2), v(3))).toSeq
    }.toSeq

  private def faceQuads(mesh: Mesh4D): Seq[Quad] =
    mesh.faces.map(f => (0 until 4).map(f(_)))

  private def range(q: Quad, axis: Int): (Float, Float) =
    (q.map(_(axis)).min, q.map(_(axis)).max)

  // Axis-aligned squares in 4D: two axes are constant, the other two span the square.
  private def overlaps(cap: Quad, face: Quad): Boolean =
    (0 until 4).forall { axis =>
      val (cMin, cMax) = range(cap, axis)
      val (fMin, fMax) = range(face, axis)
      if cMax - cMin < Tolerance then fMax - fMin < Tolerance && math.abs(cMin - fMin) < 1e-3f
      else math.min(cMax, fMax) - math.max(cMin, fMin) > Tolerance
    }

  "holeCapsBuffer" should "shrink each face to its centre third" in:
    val mesh = new Mesh4D:
      type V = 4
      override def vertices: Seq[Vector[4]] = faces.flatMap(_.asSeq).distinct
      override lazy val faces: Seq[Face4D[4]] = Seq(Face4D(
        Vector[4](0f, 0f, 0f, 0f), Vector[4](3f, 0f, 0f, 0f),
        Vector[4](3f, 3f, 0f, 0f), Vector[4](0f, 3f, 0f, 0f)))
    Mesh4DGpuFlatten.holeCapsBuffer(mesh).toSeq shouldBe
      Seq(1f, 1f, 0f, 0f, 2f, 1f, 0f, 0f, 2f, 2f, 0f, 0f, 1f, 2f, 0f, 0f)

  it should "lie inside its own source face (detector self-check)" in:
    val sponge = TesseractSponge2(1)
    val faces = faceQuads(sponge)
    capQuads(sponge).zip(faces).count { case (cap, face) => overlaps(cap, face) } shouldBe faces.size

  Seq(
    ("TesseractSponge 0 -> 1", TesseractSponge(0), TesseractSponge(1)),
    ("TesseractSponge2 0 -> 1", TesseractSponge2(0), TesseractSponge2(1)),
    ("TesseractSponge2 1 -> 2", TesseractSponge2(1), TesseractSponge2(2))
  ).foreach { case (name, current, next) =>
    it should s"leave no cap over a next-level face ($name)" in:
      val nextFaces = faceQuads(next)
      capQuads(current).count(cap => nextFaces.exists(overlaps(cap, _))) shouldBe 0
  }
