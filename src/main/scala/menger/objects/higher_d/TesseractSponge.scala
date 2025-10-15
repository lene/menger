package menger.objects.higher_d

import scala.math.abs

import menger.Const
import menger.objects.Vector


class TesseractSponge(level: Float) extends Fractal4D(level):

  require(level >= 0, "Level must be non-negative")

  lazy val faces: Seq[Face4D] = if level.toInt == 0 then Tesseract().faces else nestedFaces.flatten

  private def nestedFaces =
    for (
      xx <- -1 to 1; yy <- -1 to 1; zz <- -1 to 1; ww <- -1 to 1
      if abs(xx) + abs(yy) + abs(zz) + abs(ww) > 2
    ) yield shrunkSubSponge.map(_ + Vector[4](xx / 3f, yy / 3f, zz / 3f, ww / 3f))

  private def shrunkSubSponge: Seq[Face4D] = subSponge.map { _ / 3 }

  private def subSponge: Seq[Face4D] = TesseractSponge(level - 1).faces

  def isInSponge(point: Vector[4]): Boolean =
    if level <= 0 then
      val cubeVertices: Seq[Vector[4]] = faces.flatMap(_.asSeq)
      isInCube(point, cubeVertices)
    else
      ???

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

