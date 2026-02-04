package menger.common

enum LightType:
  case Directional, Point

sealed trait Light:
  def lightType: LightType
  def color: Color
  def intensity: Float

object Light:
  /** Directional light with parallel rays (like sunlight).
    *
    * @param direction Vector pointing TOWARD the light source position (where the light comes from).
    *                  The light rays travel in the opposite direction (-direction), shining onto the scene.
    *                  For example, direction=(1,-1,-1) places light at upper-right-back, shining toward lower-left-front.
    *                  Automatically normalized.
    * @param color Light color (RGB, each component 0.0-1.0)
    * @param intensity Light brightness multiplier (default: 1.0)
    */
  case class Directional(
    direction: Vector[3],
    color: Color = Color(1.0f, 1.0f, 1.0f),
    intensity: Float = 1.0f
  ) extends Light:
    val lightType = LightType.Directional
    require(intensity >= 0.0f, s"intensity must be non-negative, got $intensity")

  case class Point(
    position: Vector[3],
    color: Color = Color(1.0f, 1.0f, 1.0f),
    intensity: Float = 1.0f
  ) extends Light:
    val lightType = LightType.Point
    require(intensity >= 0.0f, s"intensity must be non-negative, got $intensity")
