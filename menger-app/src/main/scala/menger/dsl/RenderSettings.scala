package menger.dsl

import menger.common.RenderConfig

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
  * @param maxRayDepth Maximum ray bounce depth (1 to RenderLimits.MaxRayDepth)
  */
case class RenderSettings(
  shadows: Boolean = true,
  transparentShadows: Boolean = false,
  antialiasing: Boolean = false,
  aaMaxDepth: Int = 2,
  aaThreshold: Float = 0.1f,
  maxRayDepth: Option[Int] = None,
  // Sprint 23.5: multi-frame accumulation for noise reduction.
  accumulation: Int = 1,
  denoise: DenoiseMode = DenoiseMode.Off,
):
  require(aaMaxDepth >= 1 && aaMaxDepth <= 4, s"aaMaxDepth must be 1-4, got $aaMaxDepth")
  require(aaThreshold >= 0.0f && aaThreshold <= 1.0f, s"aaThreshold must be 0.0-1.0, got $aaThreshold")
  maxRayDepth.foreach(d => require(
    d >= 1 && d <= RenderConfig.Default.maxRayDepth,
    s"maxRayDepth must be 1-${RenderConfig.Default.maxRayDepth}, got $d"
  ))
  require(accumulation >= 1, s"RenderSettings.accumulation must be ≥ 1, got $accumulation")

  def toRenderConfig: RenderConfig = RenderConfig(
    shadows = shadows,
    transparentShadows = transparentShadows,
    antialiasing = antialiasing,
    aaMaxDepth = aaMaxDepth,
    aaThreshold = aaThreshold,
    maxRayDepth = maxRayDepth.getOrElse(RenderConfig.Default.maxRayDepth)
  )

object RenderSettings:
  val Default: RenderSettings = RenderSettings()

  val HighQuality: RenderSettings = RenderSettings(
    shadows = true,
    antialiasing = true,
    aaMaxDepth = 3,
    aaThreshold = 0.05f
  )
