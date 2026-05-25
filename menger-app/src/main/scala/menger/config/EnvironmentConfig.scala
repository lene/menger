package menger.config

import menger.common.Axis
import menger.common.Color
import menger.common.FogConfig
import menger.common.Light
import menger.common.PlaneSpec

/**
 * Environment configuration for lighting and ground planes.
 *
 * @param planes list of ground/wall planes (up to 4 simultaneous planes)
 * @param lights list of light sources (empty = use default lighting)
 * @param background optional background color
 * @param iblEnabled whether to use the env map as a diffuse IBL light source
 * @param iblStrength IBL light multiplier (1.0 = physically neutral)
 * @param iblSamples number of IBL shadow rays per hit (more = less noise)
 */
case class EnvironmentConfig(
  planes: List[PlaneConfig] = List.empty,
  lights: List[Light] = List.empty,
  background: Option[Color] = None,
  envMap: Option[String] = None,
  fog: Option[FogConfig] = None,
  iblEnabled: Boolean = false,
  iblStrength: Float = 1.0f,
  iblSamples: Int = 1
)

object EnvironmentConfig:
  /**
   * Default configuration: no planes, default lighting
   */
  val Default: EnvironmentConfig = EnvironmentConfig(
    planes = List.empty,
    lights = List.empty
  )

  /**
   * Configuration with a single gray-checker plane at Y=-2.
   */
  val WithPlane: EnvironmentConfig = EnvironmentConfig(
    planes = List(PlaneConfig(PlaneSpec(Axis.Y, positive = true, -2f), colorSpec = None)),
    lights = List.empty
  )
