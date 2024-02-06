package menger.objects

import menger.objects.Direction.{X, Y, negX, negY}

import scala.math.abs


case class Face(xCen: Float, yCen: Float, zCen: Float, scale: Float, normal: Direction):
  def subdivide(): Seq[Face] =
    val unrotatedSubFaces = for(x <- -1 to 1; y <- -1 to 1 if abs(x) + abs(y) > 0)
      yield Face(xCen + x * scale / 3f, yCen + y * scale / 3f, zCen, scale / 3f, normal)
    val rotatedSubFaces = for(
      (x, y, axis) <- Seq((0f, -1/3f, negX), (1/3f, 0f, negY), (0f, 1/3f, X), (-1/3f, 0f, Y))
    )
      yield Face(
        xCen + x * scale, yCen + y * scale, zCen - scale / 3f, scale / 3f, normal.rotate90(axis)
      )
    unrotatedSubFaces ++ rotatedSubFaces

