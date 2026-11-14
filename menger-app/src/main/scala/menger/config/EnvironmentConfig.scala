package menger.config

import menger.common.Color
import menger.common.FogConfig
import menger.common.Light
import menger.common.Vector
import menger.video.EnvMapVideo

/**
 * Environment configuration for lighting and ground planes.
 *
 * @param planes list of ground/wall planes (up to 4 simultaneous planes)
 * @param lights list of light sources (empty = use default lighting)
 * @param background optional background color
 * @param envMap optional static equirectangular environment map
 * @param envMapVideo optional animated equirectangular environment map
 * @param iblEnabled whether to use the env map as a diffuse IBL light source
 * @param iblStrength IBL light multiplier (1.0 = physically neutral)
 * @param iblSamples number of IBL shadow rays per hit (more = less noise)
 * @param iblSunExclusion the sun of the env map, when a scene light stands for it: image-based
 *                        lighting leaves that cone out, so the sun is not counted twice
 */
case class EnvironmentConfig(
  planes: List[PlaneConfig] = List.empty,
  lights: List[Light] = List.empty,
  background: Option[Color] = None,
  envMap: Option[String] = None,
  envMapVideo: Option[EnvMapVideo] = None,
  fog: Option[FogConfig] = None,
  iblEnabled: Boolean = false,
  iblStrength: Float = 1.0f,
  iblSamples: Int = 1,
  iblSunExclusion: Option[IblSunExclusion] = None
)

/** The cone of the env map that image-based lighting leaves out because a scene light stands
  * for it (`OptiXRenderer.setIBLSunExclusion`).
  *
  * @param directionToSun unit vector from the scene to the sun, in the map's frame
  * @param halfAngleDegrees half-angle of the cone
  */
case class IblSunExclusion(directionToSun: Vector[3], halfAngleDegrees: Float)

object EnvironmentConfig:
  /**
   * Default configuration: no planes, default lighting
   */
  val Default: EnvironmentConfig = EnvironmentConfig(
    planes = List.empty,
    lights = List.empty
  )
