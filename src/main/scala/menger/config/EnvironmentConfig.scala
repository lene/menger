package menger.config

import menger.Axis
import menger.LightSpec
import menger.PlaneColorSpec
import menger.PlaneSpec

/**
 * Environment configuration for lighting and ground plane.
 *
 * @param plane ground plane specification
 * @param planeColor optional checkered or solid color for the plane
 * @param lights list of light sources (default: single directional light)
 */
case class EnvironmentConfig(
  plane: PlaneSpec,
  planeColor: Option[PlaneColorSpec] = None,
  lights: Option[List[LightSpec]] = None
)

object EnvironmentConfig:
  /**
   * Default configuration: no plane, default lighting
   */
  val Default: EnvironmentConfig = EnvironmentConfig(
    plane = PlaneSpec(Axis.X, positive = false, 0f),
    planeColor = None,
    lights = None
  )

  /**
   * Configuration with plane at Y=-2 and default lighting.
   */
  val WithPlane: EnvironmentConfig = EnvironmentConfig(
    plane = PlaneSpec(Axis.Y, positive = true, -2f),
    planeColor = None,
    lights = None
  )
