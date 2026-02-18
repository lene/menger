package menger.dsl

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
