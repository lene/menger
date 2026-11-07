package menger.dsl

import menger.common.{AreaLightShape => CommonAreaLightShape}
import menger.common.{Light => CommonLight}

/** Base trait for DSL light types */
sealed trait Light:
  def toCommonLight: CommonLight

/** Directional light with parallel rays (like sunlight).
  *
  * @param direction The direction the light travels: (0f, -1f, 0f) shines straight down,
  *                  (1f, -1f, -1f) shines down and away from a camera on the +z side.
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
  FieldConstraints.check("AreaLight", "shadowSamples", shadowSamples)

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

/** The sun (or any distant disk-shaped key light): an area light on the sun's direction, sized by
  * its angular diameter (menger#158). Lights the origin like `Directional` of the same
  * `intensity`, whatever the `distance`; `distance` only sets the penumbra width.
  *
  * Only `lights(0)` emits caustic photons, so put the `Sun` first in a scene with caustics.
  * `HdriSun` (menger.tools) reads direction, diameter and colour off an HDRI.
  *
  * @param direction The direction the light travels, as `Directional.direction`; need not be
  *                  normalized, must not be zero
  * @param diameterDegrees Apparent diameter of the disk in degrees (0 to 10; the sun is 0.53)
  * @param distance Distance of the disk from the origin along the light's way (default: 50)
  * @param color Light color (default: white)
  * @param intensity Brightness, as for `Directional` (default: 1.0)
  */
case class Sun(
  direction: Vec3,
  diameterDegrees: Float = Sun.ApparentDiameterDegrees,
  distance: Float = Sun.DefaultDistance,
  color: Color = Color.White,
  intensity: Float = 1.0f
) extends Light:
  require(direction.magnitude > 0f, "Sun.direction must not be the zero vector")
  require(diameterDegrees > 0f, s"Sun.diameterDegrees must be positive, got $diameterDegrees")
  FieldConstraints.check("Sun", "diameterDegrees", diameterDegrees)
  require(distance > 0f, s"Sun.distance must be positive, got $distance")
  require(intensity >= 0f, s"Sun.intensity must be non-negative, got $intensity")

  /** The area light the renderer sees. Shading uses the disk centre with 1 / (1 + d^2), so the
    * intensity is scaled by (1 + d^2) to keep the light at the origin independent of the
    * distance. */
  def toAreaLight: AreaLight =
    val travel = direction.normalize
    AreaLight(
      position = travel * -distance,
      normal = travel,
      radius = distance * math.tan(math.toRadians(diameterDegrees.toDouble) / 2).toFloat,
      intensity = intensity * (1f + distance * distance),
      color = color,
      shadowSamples = Sun.ShadowSamples
    )

  def toCommonLight: CommonLight = toAreaLight.toCommonLight

object Sun:
  val ApparentDiameterDegrees: Float = 0.53f
  val DefaultDistance: Float = 50f
  val ShadowSamples: Int = 4
  val MaxDiameterDegrees: Float = 10f
