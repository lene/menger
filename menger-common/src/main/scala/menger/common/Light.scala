package menger.common

enum AreaLightShape(val id: Int):
  case Disk extends AreaLightShape(0)

enum LightType:
  case Directional, Point, Area

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

  case class Area(
    position: Vector[3],
    normal: Vector[3],
    radius: Float,
    shape: AreaLightShape = AreaLightShape.Disk,
    color: Color = Color(1.0f, 1.0f, 1.0f),
    intensity: Float = 1.0f,
    shadowSamples: Int = 4
  ) extends Light:
    val lightType = LightType.Area
    require(intensity >= 0.0f, s"intensity must be non-negative, got $intensity")
    require(radius > 0.0f, s"radius must be positive, got $radius")
    require(shadowSamples >= 1 && shadowSamples <= 16, s"shadowSamples must be 1-16, got $shadowSamples")
