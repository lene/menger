package menger.objects

import scala.annotation.targetName

enum Direction(val x: Byte, val y: Byte, val z: Byte):
  case X extends Direction(1, 0, 0)
  case Y extends Direction(0, 1, 0)
  case Z extends Direction(0, 0, 1)
  case negX extends Direction(-1, 0, 0)
  case negY extends Direction(0, -1, 0)
  case negZ extends Direction(0, 0, -1)

  @targetName("-")
  def unary_- : Direction = Direction((-x).toByte, (-y).toByte, (-z).toByte)

  def rotate90(D: Direction): Direction =
    val (rotX, rotY, rotZ) = getRotatedVector(D)
    Direction(rotX, rotY, rotZ)

  private def getRotatedVector(direction: Direction): (Byte, Byte, Byte) =
    (direction.x, direction.y, direction.z) match
      case (1, 0, 0) => (x, (-z).toByte, y)
      case (0, 1, 0) => (z, y, (-x).toByte)
      case (0, 0, 1) => ((-y).toByte, x, z)
      case (-1, 0, 0) => (x, z, (-y).toByte)
      case (0, -1, 0) => ((-z).toByte, y, x)
      case (0, 0, -1) => (y, (-x).toByte, z)

object Direction:
  def apply(x: Byte, y: Byte, z: Byte): Direction =
    (x, y, z) match
      case (1, 0, 0) => Direction.X
      case (0, 1, 0) => Direction.Y
      case (0, 0, 1) => Direction.Z
      case (-1, 0, 0) => Direction.negX
      case (0, -1, 0) => Direction.negY
      case (0, 0, -1) => Direction.negZ
      case _ => throw new IllegalArgumentException("Invalid direction ($x, $y, $z)")
