package menger.dsl

import menger.common.RenderConfig

/** Rendering quality settings for a DSL scene.
  *
  * These settings override CLI defaults when set in a scene file.
  * CLI flags take precedence over DSL values when explicitly supplied.
  *
  * @param shadows Enable shadow rays for realistic shadows
  * @param transparentShadows Colored tinting through transparent objects (requires shadows):
  *   `Auto` (default) when a transparent object is coloured, or `true`/`false`
  * @param antialiasing Enable recursive adaptive antialiasing
  * @param aaMaxDepth Maximum AA recursion depth (1-4)
  * @param aaThreshold AA edge detection threshold (0.0-1.0)
  * @param maxRayDepth Maximum ray bounce depth (1 to RenderLimits.MaxRayDepth)
  */
case class RenderSettings(
  shadows: Boolean = true,
  transparentShadows: TransparentShadows = TransparentShadows.Auto,
  antialiasing: Boolean = false,
  aaMaxDepth: Int = 2,
  aaThreshold: Float = 0.1f,
  maxRayDepth: Option[Int] = None,
  // Sprint 23.5: multi-frame accumulation for noise reduction.
  accumulation: Int = 1,
  denoise: DenoiseMode = DenoiseMode.Off,
):
  FieldConstraints.check("RenderSettings", "aaMaxDepth", aaMaxDepth)
  FieldConstraints.check("RenderSettings", "aaThreshold", aaThreshold)
  maxRayDepth.foreach(FieldConstraints.check("RenderSettings", "maxRayDepth", _))
  FieldConstraints.check("RenderSettings", "accumulation", accumulation)

  def toRenderConfig: RenderConfig = toRenderConfigFor(colouredTransparent = false)

  /** `colouredTransparent`: the scene has a coloured transparent object, which `Auto` shows. */
  def toRenderConfigFor(colouredTransparent: Boolean): RenderConfig = RenderConfig(
    shadows = shadows,
    transparentShadows = transparentShadows.enabled(colouredTransparent),
    antialiasing = antialiasing,
    aaMaxDepth = aaMaxDepth,
    aaThreshold = aaThreshold,
    maxRayDepth = maxRayDepth.getOrElse(RenderConfig.Default.maxRayDepth)
  )

/** Whether a transparent object's shadow takes the colour seen through it. `Auto` turns it on
  * when the scene has a coloured or procedurally coloured transparent object: the tester expected
  * a coloured shadow from coloured glass without asking for it (usability review 2026-10,
  * session 5, F111, menger#128; decided by user 2026-10-22: "Auto-on + fix chord"). A Boolean
  * is `On` or `Off`, so `transparentShadows = true` reads as before. */
enum TransparentShadows:
  case On, Off, Auto

  def enabled(colouredTransparent: Boolean): Boolean = this match
    case On   => true
    case Off  => false
    case Auto => colouredTransparent

object TransparentShadows:
  given Conversion[Boolean, TransparentShadows] = b => if b then On else Off

object RenderSettings:
  val Default: RenderSettings = RenderSettings()

  val HighQuality: RenderSettings = RenderSettings(
    shadows = true,
    antialiasing = true,
    aaMaxDepth = 3,
    aaThreshold = 0.05f
  )
