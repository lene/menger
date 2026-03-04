package menger.config

import menger.cli.Axis
import menger.cli.LightSpec
import menger.cli.PlaneConfig
import menger.cli.PlaneSpec
import menger.common.Color

/**
 * Environment configuration for lighting and ground planes.
 *
 * @param planes list of ground/wall planes (up to 4 simultaneous planes)
 * @param lights list of light sources (empty = use default lighting)
 * @param background optional background color
 */
case class EnvironmentConfig(
  planes: List[PlaneConfig] = List.empty,
  lights: List[LightSpec] = List.empty,
  background: Option[Color] = None
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
