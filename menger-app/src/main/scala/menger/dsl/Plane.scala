package menger.dsl

import menger.cli.{Axis, PlaneColorSpec, PlaneSpec}

/** Axis position specification for planes.
  *
  * Created by axis helpers: `Y at -2` produces `AxisPosition(Axis.Y, false, -2f)`
  *
  * @param axis The axis perpendicular to the plane (X, Y, or Z)
  * @param positive Whether the plane is on the positive side of origin
  * @param value The position along the axis
  */
case class AxisPosition(axis: Axis, positive: Boolean, value: Float)

/** Axis helper for creating plane positions with natural syntax.
  *
  * Example: `Y at -2` creates a horizontal plane at y = -2
  */
sealed trait AxisHelper:
  def axis: Axis

  /** Create an axis position at the given value.
    *
    * @param value Position along the axis
    * @return AxisPosition with positive flag set based on value sign
    */
  infix def at(value: Float): AxisPosition = AxisPosition(axis, value >= 0, value)

  /** Create an axis position at the given integer value.
    *
    * @param value Position along the axis (converted to Float)
    * @return AxisPosition with positive flag set based on value sign
    */
  infix def at(value: Int): AxisPosition = at(value.toFloat)

  /** Create an axis position at the given double value.
    *
    * @param value Position along the axis (converted to Float)
    * @return AxisPosition with positive flag set based on value sign
    */
  infix def at(value: Double): AxisPosition = at(value.toFloat)

/** X-axis helper for plane definitions. */
case object X extends AxisHelper:
  val axis: Axis = Axis.X

/** Y-axis helper for plane definitions (most common for floor planes). */
case object Y extends AxisHelper:
  val axis: Axis = Axis.Y

/** Z-axis helper for plane definitions. */
case object Z extends AxisHelper:
  val axis: Axis = Axis.Z

/** Plane definition for scene floor/walls.
  *
  * Planes are infinite surfaces perpendicular to one of the coordinate axes.
  * They can be solid-colored or checkered.
  *
  * Examples:
  * {{{
  *   // Solid gray floor at y = -2
  *   Plane(Y at -2, color = "#808080")
  *
  *   // Checkered floor (white and black)
  *   Plane(Y at -2, checkered = (Color.White, Color.Black))
  *
  *   // Using hex colors for checkered pattern
  *   Plane(Y at -2, checkered = ("#FFFFFF", "#000000"))
  * }}}
  *
  * @param axisPosition Position and orientation of the plane
  * @param color Optional solid color for the plane
  * @param checkered Optional pair of colors for checkered pattern
  */
case class Plane(
  axisPosition: AxisPosition,
  color: Option[Color] = None,
  checkered: Option[(Color, Color)] = None
):
  require(
    color.isDefined || checkered.isDefined,
    "Plane must have either color or checkered pattern defined"
  )
  require(
    !(color.isDefined && checkered.isDefined),
    "Plane cannot have both color and checkered pattern"
  )

  /** Convert to PlaneSpec for rendering. */
  def toPlaneSpec: PlaneSpec =
    PlaneSpec(axisPosition.axis, axisPosition.positive, axisPosition.value)

  /** Convert to PlaneColorSpec for rendering. */
  def toPlaneColorSpec: PlaneColorSpec =
    if color.isDefined then
      PlaneColorSpec(color.get.toCommonColor, None)
    else
      val (c1, c2) = checkered.get  // Safe due to require checks
      PlaneColorSpec(c1.toCommonColor, Some(c2.toCommonColor))

object Plane:
  /** Create a solid-colored plane.
    *
    * @param axisPosition Position specification (e.g., `Y at -2`)
    * @param color Solid color
    * @return Plane with solid color
    */
  def apply(axisPosition: AxisPosition, color: Color): Plane =
    Plane(axisPosition, Some(color), None)

  /** Create a solid-colored plane from hex string.
    *
    * @param axisPosition Position specification (e.g., `Y at -2`)
    * @param colorHex Hex color string (e.g., "#808080")
    * @return Plane with solid color
    */
  def apply(axisPosition: AxisPosition, colorHex: String): Plane =
    Plane(axisPosition, Some(Color(colorHex)), None)

  /** Create a checkered plane.
    *
    * @param axisPosition Position specification (e.g., `Y at -2`)
    * @param colors Pair of colors for checkered pattern
    * @return Plane with checkered pattern
    */
  def checkered(axisPosition: AxisPosition, colors: (Color, Color)): Plane =
    Plane(axisPosition, None, Some(colors))

  /** Create a checkered plane from hex strings.
    *
    * @param axisPosition Position specification (e.g., `Y at -2`)
    * @param color1Hex First hex color string
    * @param color2Hex Second hex color string
    * @return Plane with checkered pattern
    */
  def checkered(axisPosition: AxisPosition, color1Hex: String, color2Hex: String): Plane =
    Plane(axisPosition, None, Some((Color(color1Hex), Color(color2Hex))))
