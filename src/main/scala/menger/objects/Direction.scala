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
    val rot = getRotatedVector(D)
    Direction(rot)

  lazy val fold1: Direction =
    /**
     *  X => negY
     *  Y => negZ
     *  Z => negX
     *  negX => Y
     *  negY => Z
     *  negZ => X
     */
    val newOrdinal = (this.ordinal + 1) % 3
    -Direction.fromOrdinal(newOrdinal)

  lazy val fold2: Direction =
    /**
     *  X => negZ
     *  Y => negX
     *  Z => negY
     *  negX => Z
     *  negY => X
     *  negZ => Y
     */
    val newOrdinal = (this.ordinal + 2) % 3
    -Direction.fromOrdinal(newOrdinal)

  lazy val fold3: Direction = -fold1
  lazy val fold4: Direction = -fold2

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

  def apply(xyz: (Byte, Byte, Byte)): Direction = apply(xyz._1, xyz._2, xyz._3)