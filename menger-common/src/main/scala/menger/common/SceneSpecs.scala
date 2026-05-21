package menger.common

/** Axis identifier for plane definitions. */
enum Axis:
  case X, Y, Z

/** Position of a plane along an axis. */
case class PlaneSpec(axis: Axis, positive: Boolean, value: Float)

/** Color specification for a plane (solid or checkered). */
case class PlaneColorSpec(color1: Color, color2: Option[Color]):
  def isSolid: Boolean    = color2.isEmpty
  def isCheckered: Boolean = color2.isDefined

/** Fog rendering parameters. */
case class FogSpec(density: Float, color: Color = Color(0.8f, 0.8f, 0.9f))
