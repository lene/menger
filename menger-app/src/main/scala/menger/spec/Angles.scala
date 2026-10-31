package menger.spec

import scala.language.implicitConversions

/** An angle in degrees. A plain `Float` converts implicitly, so scene text that passes `30f`
  * keeps compiling; the type is what stops a maintainer handing degrees to code that wants
  * radians, and what lets the manifest state the unit of a field (menger#95). */
opaque type Degrees = Float

object Degrees:
  val Zero: Degrees = 0f

  def apply(value: Float): Degrees = value

  given Conversion[Float, Degrees] = apply(_)

  extension (angle: Degrees)
    def value: Float = angle
    def toRadians: Radians = Radians(math.toRadians(angle.toDouble).toFloat)
    def +(other: Degrees): Degrees = angle + other
    def -(other: Degrees): Degrees = angle - other
    def unary_- : Degrees = -angle
    def *(factor: Float): Degrees = angle * factor

/** An angle in radians; see [[Degrees]]. */
opaque type Radians = Float

object Radians:
  val Zero: Radians = 0f

  def apply(value: Float): Radians = value

  given Conversion[Float, Radians] = apply(_)

  extension (angle: Radians)
    def value: Float = angle
    def toDegrees: Degrees = Degrees(math.toDegrees(angle.toDouble).toFloat)
    def +(other: Radians): Radians = angle + other
    def -(other: Radians): Radians = angle - other
    def unary_- : Radians = -angle
    def *(factor: Float): Radians = angle * factor

extension (value: Float)
  def deg: Degrees = Degrees(value)
  def rad: Radians = Radians(value)
