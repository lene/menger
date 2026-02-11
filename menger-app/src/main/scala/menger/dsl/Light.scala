package menger.dsl

import menger.common.Light as CommonLight
import scala.annotation.targetName

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

object Directional:
  // Overloads for Float tuple positions
  def apply(direction: (Float, Float, Float)): Directional =
    Directional(Vec3(direction._1, direction._2, direction._3))

  @targetName("applyFloatIntensity")
  def apply(direction: (Float, Float, Float), intensity: Float): Directional =
    Directional(Vec3(direction._1, direction._2, direction._3), intensity)

  @targetName("applyFloatIntensityColor")
  def apply(direction: (Float, Float, Float), intensity: Float, color: Color): Directional =
    Directional(Vec3(direction._1, direction._2, direction._3), intensity, color)

  // Overloads for Int tuple positions
  @targetName("applyInt")
  def apply(direction: (Int, Int, Int)): Directional =
    Directional(Vec3(direction._1.toFloat, direction._2.toFloat, direction._3.toFloat))

  @targetName("applyIntIntensity")
  def apply(direction: (Int, Int, Int), intensity: Float): Directional =
    Directional(Vec3(direction._1.toFloat, direction._2.toFloat, direction._3.toFloat), intensity)

  @targetName("applyIntIntensityColor")
  def apply(direction: (Int, Int, Int), intensity: Float, color: Color): Directional =
    Directional(Vec3(direction._1.toFloat, direction._2.toFloat, direction._3.toFloat), intensity, color)

  // Overloads for Double tuple positions
  @targetName("applyDouble")
  def apply(direction: (Double, Double, Double)): Directional =
    Directional(Vec3(direction._1.toFloat, direction._2.toFloat, direction._3.toFloat))

  @targetName("applyDoubleIntensity")
  def apply(direction: (Double, Double, Double), intensity: Float): Directional =
    Directional(Vec3(direction._1.toFloat, direction._2.toFloat, direction._3.toFloat), intensity)

  @targetName("applyDoubleIntensityColor")
  def apply(direction: (Double, Double, Double), intensity: Float, color: Color): Directional =
    Directional(Vec3(direction._1.toFloat, direction._2.toFloat, direction._3.toFloat), intensity, color)

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

object Point:
  // Overloads for Float tuple positions
  def apply(position: (Float, Float, Float)): Point =
    Point(Vec3(position._1, position._2, position._3))

  @targetName("applyFloatIntensity")
  def apply(position: (Float, Float, Float), intensity: Float): Point =
    Point(Vec3(position._1, position._2, position._3), intensity)

  @targetName("applyFloatIntensityColor")
  def apply(position: (Float, Float, Float), intensity: Float, color: Color): Point =
    Point(Vec3(position._1, position._2, position._3), intensity, color)

  // Overloads for Int tuple positions
  @targetName("applyInt")
  def apply(position: (Int, Int, Int)): Point =
    Point(Vec3(position._1.toFloat, position._2.toFloat, position._3.toFloat))

  @targetName("applyIntIntensity")
  def apply(position: (Int, Int, Int), intensity: Float): Point =
    Point(Vec3(position._1.toFloat, position._2.toFloat, position._3.toFloat), intensity)

  @targetName("applyIntIntensityColor")
  def apply(position: (Int, Int, Int), intensity: Float, color: Color): Point =
    Point(Vec3(position._1.toFloat, position._2.toFloat, position._3.toFloat), intensity, color)

  // Overloads for Double tuple positions
  @targetName("applyDouble")
  def apply(position: (Double, Double, Double)): Point =
    Point(Vec3(position._1.toFloat, position._2.toFloat, position._3.toFloat))

  @targetName("applyDoubleIntensity")
  def apply(position: (Double, Double, Double), intensity: Float): Point =
    Point(Vec3(position._1.toFloat, position._2.toFloat, position._3.toFloat), intensity)

  @targetName("applyDoubleIntensityColor")
  def apply(position: (Double, Double, Double), intensity: Float, color: Color): Point =
    Point(Vec3(position._1.toFloat, position._2.toFloat, position._3.toFloat), intensity, color)
