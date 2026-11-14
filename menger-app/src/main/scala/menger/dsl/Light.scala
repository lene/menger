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
  * `intensity` at the origin; `distance` sets how uniform the light is: shading falls off as
  * 1 / (1 + r^2) from the disk, exact at the origin, so a point s units toward the light is about
  * (1 + 2s / distance) times brighter. The penumbra does not depend on it (the disk keeps its
  * angular size).
  *
  * Only `lights(0)` emits caustic photons, so put the `Sun` first in a scene with caustics.
  * `HdriSun` (menger.tools) reads direction, diameter and colour off an HDRI;
  * `Sun.fromEnvMap` takes the sun of the scene's own environment map.
  *
  * @param direction The direction the light travels, as `Directional.direction`; need not be
  *                  normalized, must not be zero
  * @param diameterDegrees Apparent diameter of the disk in degrees (0 to 10; the sun is 0.53)
  * @param distance Distance of the disk from the origin along the light's way (default: 1000)
  * @param color Light color (default: white)
  * @param intensity Brightness, as for `Directional` (default: 1.0)
  * @param measuredFromEnvMap The environment map (path under `--texture-dir`) this sun was
  *                           measured from: set by `Sun.fromEnvMap`, not by hand. When it is the
  *                           scene's `envMap` and IBL is on, the renderer leaves that map's sun
  *                           out of the image-based lighting, so the sun is not counted twice.
  */
case class Sun(
  direction: Vec3,
  diameterDegrees: Float = Sun.ApparentDiameterDegrees,
  distance: Float = Sun.DefaultDistance,
  color: Color = Color.White,
  intensity: Float = 1.0f,
  measuredFromEnvMap: Option[String] = None
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
  val DefaultDistance: Float = 1000f
  val ShadowSamples: Int = 4
  val MaxDiameterDegrees: Float = 10f

  /** Where the sun is when the environment map cannot be read (the scene validator mounts no
    * texture directory): from above, behind and to the left, white, intensity 1.
    */
  private val StandInDirection = Vec3(0.5f, -0.7f, 0.5f)

  /** The sun of the environment map at `path` (relative to `--texture-dir`), measured from the
    * `.hdr`: it travels opposite to the direction the map's sun is in, has the map's sun colour
    * and the intensity that lights a matte surface facing it as the map's sun does, and the
    * physical 0.53 degree diameter. Measured once per file.
    *
    * Use it with `envMap = Some(path)` and `ibl`: the scene then lights with this sun and the
    * renderer leaves the sun out of the map's lighting, so that it counts once. A map that is
    * not there makes a stand-in sun (white, intensity 1) in the validator and an error in a
    * render.
    */
  def fromEnvMap(path: String): Sun = fromMeasurement(path, EnvMapSunSource.measure(path))

  private[dsl] def fromMeasurement(path: String, measurement: Option[EnvMapSunMeasurement]): Sun =
    measurement.fold(Sun(direction = StandInDirection))(measured =>
      Sun(
        direction = measured.directionToSun * -1f,
        color = measured.color,
        intensity = measured.intensity,
        measuredFromEnvMap = Some(path)
      ))
