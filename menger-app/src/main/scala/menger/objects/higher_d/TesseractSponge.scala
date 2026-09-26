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
    if size == 1f then unitFaces else unitFaces.map(_ / (1f / size))
  override def cells: Seq[Cell4D] = Seq.empty

  private lazy val unitFaces: Seq[Face4D[V]] =
    if level.toInt == 0 then Tesseract().faces else nestedFaces.flatten

  private def nestedFaces =
    for (
      xx <- -1 to 1; yy <- -1 to 1; zz <- -1 to 1; ww <- -1 to 1
      if abs(xx) + abs(yy) + abs(zz) + abs(ww) > 2
    ) yield shrunkSubSponge.map(_ + Vector[4](xx / 3f, yy / 3f, zz / 3f, ww / 3f))

  private def shrunkSubSponge: Seq[Face4D[V]] = subSponge.map { _ / 3 }

  private def subSponge: Seq[Face4D[V]] = TesseractSponge(level - 1).faces

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

