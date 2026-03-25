package menger.dsl

import menger.common.{AreaLightShape => CommonAreaLightShape}
import menger.common.{Light => CommonLight}

/** Base trait for DSL light types */
sealed trait Light:
  def toCommonLight: CommonLight

/** Directional light with parallel rays (like sunlight).
  *
  * @param direction Vector pointing TOWARD the light source position (where the light comes from).
  * @param intensity Light brightness multiplier (default: 1.0)
  * @param color Light color (default: white)
  */
case class Directional(
  direction: Vec3,
  intensity: Float = 1.0f,
  color: Color = Color.White
) extends Light:
  require(intensity >= 0f, s"Intensity must be non-negative, got $intensity")

  def toCommonLight: CommonLight =
    CommonLight.Directional(
      direction = direction.toCommonVector,
      color = color.toCommonColor,
      intensity = intensity
    )

object Directional

/** Point light that radiates in all directions from a position.
  *
  * @param position Light source position
  * @param intensity Light brightness multiplier (default: 1.0)
  * @param color Light color (default: white)
  */
case class Point(
  position: Vec3,
  intensity: Float = 1.0f,
  color: Color = Color.White
) extends Light:
  require(intensity >= 0f, s"Intensity must be non-negative, got $intensity")

  def toCommonLight: CommonLight =
    CommonLight.Point(
      position = position.toCommonVector,
      color = color.toCommonColor,
      intensity = intensity
    )

object Point

/** Shape of an area light emitter. */
enum AreaLightShape:
  case Disk

/** Area light that emits from a finite disk surface, producing soft shadows.
  *
  * @param position Center of the disk in world space
  * @param normal Direction the disk faces (toward the scene), automatically normalized
  * @param radius Disk radius in world units
  * @param shape Emitter shape (currently only Disk is supported)
  * @param intensity Light brightness multiplier (default: 1.0)
  * @param color Light color (default: white)
  * @param shadowSamples Number of shadow rays per light (1-16, default: 4)
  */
case class AreaLight(
  position: Vec3,
  normal: Vec3,
  radius: Float,
  shape: AreaLightShape = AreaLightShape.Disk,
  intensity: Float = 1.0f,
  color: Color = Color.White,
  shadowSamples: Int = 4
) extends Light:
  require(intensity >= 0f, s"Intensity must be non-negative, got $intensity")
  require(radius > 0f, s"Radius must be positive, got $radius")
  require(shadowSamples >= 1 && shadowSamples <= 16, s"shadowSamples must be 1-16, got $shadowSamples")

  def toCommonLight: CommonLight =
    CommonLight.Area(
      position = position.toCommonVector,
      normal = normal.toCommonVector,
      radius = radius,
      shape = shape match { case AreaLightShape.Disk => CommonAreaLightShape.Disk },
      color = color.toCommonColor,
      intensity = intensity,
      shadowSamples = shadowSamples
    )

object AreaLight
