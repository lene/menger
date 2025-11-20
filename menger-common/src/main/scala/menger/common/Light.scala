package menger.common

enum LightType:
  case Directional, Point

sealed trait Light:
  def lightType: LightType
  def color: Vector[3]
  def intensity: Float

object Light:
  case class Directional(
    direction: Vector[3],
    color: Vector[3] = Vector[3](1.0f, 1.0f, 1.0f),
    intensity: Float = 1.0f
  ) extends Light:
    val lightType = LightType.Directional
    require(intensity >= 0.0f, s"intensity must be non-negative, got $intensity")

  case class Point(
    position: Vector[3],
    color: Vector[3] = Vector[3](1.0f, 1.0f, 1.0f),
    intensity: Float = 1.0f
  ) extends Light:
    val lightType = LightType.Point
    require(intensity >= 0.0f, s"intensity must be non-negative, got $intensity")
