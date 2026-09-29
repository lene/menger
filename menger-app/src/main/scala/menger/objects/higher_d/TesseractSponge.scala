package menger.objects.higher_d

import scala.math.abs

import menger.common.Const
import menger.common.NotYetImplementedException
import menger.common.Vector


/** @param size scale of the whole sponge; 1 matches `Tesseract(1)`. Was missing, so a DSL
  *             `TesseractSponge(size = 2.5f)` rendered at unit size (usability review 2026-09,
  *             F27). */
class TesseractSponge(level: Float, size: Float = 1f) extends Fractal4D(level):

  require(level >= 0, "Level must be non-negative")
  require(size > 0, s"Size must be positive, got $size")

  lazy val vertices: Seq[Vector[4]] = faces.flatMap(_.asSeq).distinct
  lazy val faces: Seq[Face4D[V]] =
    val unitFaces = TesseractSponge.surfaceFaces(rawUnitFaces)
    if size == 1f then unitFaces else unitFaces.map(_ / (1f / size))
  override def cells: Seq[Cell4D] = Seq.empty

  /** Every face of every sub-tesseract, shared ones included: only the complete list tells a
    * face inside the sponge from one on its surface, so the recursion passes it up unfiltered. */
  private lazy val rawUnitFaces: Seq[Face4D[V]] =
    if level.toInt == 0 then Tesseract().faces else nestedFaces.flatten

  private def nestedFaces =
    for (
      xx <- -1 to 1; yy <- -1 to 1; zz <- -1 to 1; ww <- -1 to 1
      if abs(xx) + abs(yy) + abs(zz) + abs(ww) > 2
    ) yield shrunkSubSponge.map(_ + Vector[4](xx / 3f, yy / 3f, zz / 3f, ww / 3f))

  private def shrunkSubSponge: Seq[Face4D[V]] = subSponge.map { _ / 3 }

  private def subSponge: Seq[Face4D[V]] = TesseractSponge(level - 1).rawUnitFaces

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  def isInSponge(point: Vector[4]): Boolean =
    if level <= 0 then
      val cubeVertices: Seq[Vector[4]] = faces.flatMap(_.asSeq)
      isInCube(point, cubeVertices)
    else
      throw NotYetImplementedException(s"isInSponge for level $level > 0")

  private[higher_d] def isInCube(point: Vector[4], cubeVertices: Seq[Vector[4]]): Boolean =
    // Get unique vertices (faces may share vertices)
    val uniqueVertices = cubeVertices.distinct

    require(uniqueVertices.size == 16, s"A 4D cube must have exactly 16 vertices, got ${uniqueVertices.size}")

    // Validate that edges are parallel to axes: each dimension should have exactly 2 distinct values
    (0 to 3).foreach { i =>
      val distinctValues = uniqueVertices.map(_(i)).distinct.size
      require(distinctValues == 2,
        s"Cube edges must be parallel to axes: dimension $i has $distinctValues distinct values, expected 2")
    }

    val minBound: Vector[4] = Vector[4](
      (0 to 3).map(i => uniqueVertices.map(_(i)).min) *
    )
    val maxBound: Vector[4] = Vector[4](
      (0 to 3).map(i => uniqueVertices.map(_(i)).max) *
    )
    (0 to 3).forall(i =>
        point(i) >= minBound(i) - Const.epsilon &&
          point(i) <= maxBound(i) + Const.epsilon
      )

object TesseractSponge:

  // A 2-face of the 4D grid borders four hypercubes: the two axes it doesn't span, each +/-.
  private val InteriorMultiplicity = 4
  private val KeyScale = 1e4f

  private def key(face: Face4D[?]): Seq[(Int, Int, Int, Int)] =
    face.asSeq.map { v =>
      (math.round(v(0) * KeyScale), math.round(v(1) * KeyScale),
        math.round(v(2) * KeyScale), math.round(v(3) * KeyScale))
    }.sorted

  /** The sponge's surface from the faces of all its sub-tesseracts: a face emitted by all four
    * hypercubes around it is inside the sponge and dropped, and every other face is kept once.
    * Keeping every copy put 2-4 coincident faces wherever sub-tesseracts touch, and glass
    * refracted at each of them (usability review 2026-09, session 2, F55). */
  private[higher_d] def surfaceFaces[F <: Face4D[?]](raw: Seq[F]): Seq[F] =
    val copies = raw.groupMapReduce(key)(_ => 1)(_ + _)
    raw.filter(face => copies(key(face)) < InteriorMultiplicity).distinctBy(key)
