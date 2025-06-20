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
  private[objects] def unary_- : Direction = Direction((-x).toByte, (-y).toByte, (-z).toByte)

  def rotate90(d: Direction): Direction = Direction(getRotatedVector(d))

  lazy val fold1: Direction =
    /**
     *  Something something cross product, I dunno but it works
     *  X => -Y
     *  Y => -Z
     *  Z => -X
     *  -X => Y
     *  -Y => Z
     *  -Z => X
     */
    val foldedOrdinal = (this.abs.ordinal + 1) % 3
    -Direction.fromOrdinal(foldedOrdinal)

  lazy val fold2: Direction =
    /**
     *  As above but cross product with a different axis I guess
     *  X => -Z
     *  Y => -X
     *  Z => -Y
     *  -X => Z
     *  -Y => X
     *  -Z => Y
     */
    val foldedOrdinal = (this.abs.ordinal + 2) % 3
    -Direction.fromOrdinal(foldedOrdinal)

  lazy val sign: Byte  = (x + y + z).toByte

  private def getRotatedVector(direction: Direction): (Byte, Byte, Byte) =
    /** rotate 90 degrees around axis `direction` */
    direction match
      case Direction.X => (x, (-z).toByte, y)
      case Direction.Y => (z, y, (-x).toByte)
      case Direction.Z => ((-y).toByte, x, z)
      case Direction.negX => (x, z, (-y).toByte)
      case Direction.negY => ((-z).toByte, y, x)
      case Direction.negZ => (y, (-x).toByte, z)

  private[objects] def abs: Direction = Direction(x.abs, y.abs, z.abs)

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

  def apply(xyz: (Byte, Byte, Byte)): Direction = apply.tupled(xyz)
