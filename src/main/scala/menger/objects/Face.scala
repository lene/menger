package menger.objects

import com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo
import menger.objects.Direction.{X, Y, Z, negX, negY, negZ}

import scala.math.abs


case class Face(xCen: Float, yCen: Float, zCen: Float, scale: Float, normal: Direction):
  def subdivide(): Seq[Face] =
    unrotatedSubFaces ++ rotatedSubFaces

  lazy val vertices: (VertexInfo, VertexInfo, VertexInfo, VertexInfo) =
    val half = scale / 2
    normal match
      case X | Direction.negX => createVertices((0, -half, -half), (0, -half, half), (0, half, half), (0, half, -half))
      case Y | Direction.negY => createVertices((-half, 0, -half), (half, 0, -half), (half, 0, half), (-half, 0, half))
      case Z | Direction.negZ => createVertices((-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0))

  private def createVertices(offsets: (Float, Float, Float)*): (VertexInfo, VertexInfo, VertexInfo, VertexInfo) =
    offsets.map { case (dx, dy, dz) => VertexInfo().setPos(xCen + dx, yCen + dy, zCen + dz) } match
      case Seq(v1, v2, v3, v4) => (v1, v2, v3, v4)
  
  private lazy val unrotatedSubFaces: Seq[Face] =
    for (x <- -1 to 1; y <- -1 to 1 if abs(x) + abs(y) > 0)
      yield Face(runningCoordinates(x/3f, y/3f, scale), scale / 3f, normal)

  private lazy val rotatedSubFaces: Seq[Face] =
    for(
      (x, y, axis) <- Seq(
        (0f, -1/3f, normal.fold1), (1/3f, 0f, normal.fold2),
        (0f, 1/3f, -normal.fold1), (-1/3f, 0f, -normal.fold2)
      )
    ) yield Face(
      runningCoordinatesShifted(x, y, scale, -scale/3f), scale / 3f, normal.rotate90(axis)
    )

  private def runningCoordinates(x1: Float, x2: Float, add: Float): (Float, Float, Float) =
    normal match
      case X | Direction.negX => (xCen, yCen + x1 * add, zCen + x2 * add)
      case Y | Direction.negY => (xCen + x2 * add, yCen, zCen + x1 * add)
      case Z | Direction.negZ => (xCen + x1 * add, yCen + x2 * add, zCen)

  private def runningCoordinatesShifted(
    x1: Float, x2: Float, add: Float, shift: Float
  ): (Float, Float, Float) =
    normal match
      case X | Direction.negX => (xCen + shift / 2 * normal.sign, yCen + x1 * add / 2, zCen + x2 * add / 2)
      case Y | Direction.negY => (xCen + x2 * add / 2, yCen + shift / 2 * normal.sign, zCen + x1 * add / 2)
      case Z | Direction.negZ => (xCen + x1 * add / 2, yCen + x2 * add / 2, zCen + shift / 2 * normal.sign)

object Face:
  def apply(xyzCen: (Float, Float, Float), scale: Float, normal: Direction): Face =
    Face(xyzCen._1, xyzCen._2, xyzCen._3, scale, normal)