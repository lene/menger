package menger.objects

import menger.objects.Direction.{X, Y, Z, negX, negY, negZ}

import scala.math.abs


case class Face(xCen: Float, yCen: Float, zCen: Float, scale: Float, normal: Direction):
  def subdivide(): Seq[Face] =
    unrotatedSubFaces ++ rotatedSubFaces

  private lazy val unrotatedSubFaces: Seq[Face] =
    for (x <- -1 to 1; y <- -1 to 1 if abs(x) + abs(y) > 0)
      yield Face(runningCoordinates(x, y, scale), scale / 3f, normal)

  private lazy val rotatedSubFaces: Seq[Face] =
    for(
      (x, y, axis) <- Seq(
        (0f, -1/3f, normal.fold1), (1/3f, 0f, normal.fold2),
        (0f, 1/3f, -normal.fold1), (-1/3f, 0f, -normal.fold2)
      )
    ) yield Face(
      runningCoordinatesShifted(x, y, scale, -scale/3f), scale / 3f, normal.rotate90(axis)
    )

  private def runningCoordinates(x1: Int, x2: Int, add: Float): (Float, Float, Float) =
    normal match
      case X | Direction.negX => (xCen, yCen + x1 * add, zCen + x2 * add)
      case Y | Direction.negY => (xCen + x2 * add, yCen, zCen + x1 * add)
      case Z | Direction.negZ => (xCen + x1 * add, yCen + x2 * add, zCen)

  private def runningCoordinatesShifted(
    x1: Float, x2: Float, add: Float, shift: Float
  ): (Float, Float, Float) =
    normal match
      case X | Direction.negX => (xCen + shift * normal.sign, yCen + x1 * add, zCen + x2 * add)
      case Y | Direction.negY => (xCen + x2 * add, yCen + shift * normal.sign, zCen + x1 * add)
      case Z | Direction.negZ => (xCen + x1 * add, yCen + x2 * add, zCen + shift * normal.sign)

object Face:
  def apply(xyzCen: (Float, Float, Float), scale: Float, normal: Direction): Face =
    Face(xyzCen._1, xyzCen._2, xyzCen._3, scale, normal)