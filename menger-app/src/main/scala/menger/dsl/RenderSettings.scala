package menger.dsl

import menger.optix.RenderConfig

/** Rendering quality settings for a DSL scene.
  *
  * These settings override CLI defaults when set in a scene file.
  * CLI flags take precedence over DSL values when explicitly supplied.
  *
  * @param shadows Enable shadow rays for realistic shadows
  * @param transparentShadows Enable colored tinting through transparent objects (requires shadows)
  * @param antialiasing Enable recursive adaptive antialiasing
  * @param aaMaxDepth Maximum AA recursion depth (1-4)
  * @param aaThreshold AA edge detection threshold (0.0-1.0)
  * @param maxRayDepth Maximum ray bounce depth — NOT YET IMPLEMENTED, must be None
  */
case class RenderSettings(
  shadows: Boolean = false,
  transparentShadows: Boolean = false,
  antialiasing: Boolean = false,
  aaMaxDepth: Int = 2,
  aaThreshold: Float = 0.1f,
  maxRayDepth: Option[Int] = None
):
  require(aaMaxDepth >= 1 && aaMaxDepth <= 4, s"aaMaxDepth must be 1-4, got $aaMaxDepth")
  require(aaThreshold >= 0.0f && aaThreshold <= 1.0f, s"aaThreshold must be 0.0-1.0, got $aaThreshold")
  maxRayDepth.foreach(_ => failMaxRayDepth())

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  private def failMaxRayDepth(): Nothing =
    throw new NotImplementedError("maxRayDepth is not yet implemented in the OptiX renderer")

  def toRenderConfig: RenderConfig = RenderConfig(
    shadows = shadows,
    transparentShadows = transparentShadows,
    antialiasing = antialiasing,
    aaMaxDepth = aaMaxDepth,
    aaThreshold = aaThreshold
  )

object RenderSettings:
  val Default: RenderSettings = RenderSettings()

  val HighQuality: RenderSettings = RenderSettings(
    shadows = true,
    antialiasing = true,
    aaMaxDepth = 3,
    aaThreshold = 0.05f
  )
