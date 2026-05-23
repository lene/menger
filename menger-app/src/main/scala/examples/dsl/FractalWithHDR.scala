package examples.dsl

import scala.language.implicitConversions

import menger.Projection4DSpec
import menger.dsl._

/**
 * Animated 4D Menger sponge (glass) with HDR environment map and Reinhard tone mapping.
 *
 * Animation (t: 0 → 3):
 *   t ∈ [0,1]: level 1→2, XW rotation sweeps 0°→90°
 *   t ∈ [1,2]: level 2→3, YW rotation sweeps 0°→90° (XW held at 90°)
 *   t ∈ [2,3]: level 3→4, ZW rotation sweeps 0°→90° (XW+YW held at 90°)
 *
 * The glass material reveals the 4D projection geometry through refraction.
 * Cliffside HDR (high-contrast midday sun) makes tone mapping visible.
 *
 * Usage:
 *   --scene examples.dsl.FractalWithHDR --texture-dir menger-app/src/test/resources/ \
 *     --t 1.5 --max-instances 5000
 *   --scene examples.dsl.FractalWithHDR --texture-dir menger-app/src/test/resources/ \
 *     --frames 90 --start-t 0 --end-t 3 --max-instances 5000 \
 *     --save-name fractal_hdr_%04d.png
 */
object FractalWithHDR:
  private val EyeW    = 3.0f
  private val ScreenW = 1.5f

  def scene(t: Float): Scene =
    val tClamped = math.max(0f, math.min(t, 3f))
    val level = 1f + tClamped
    val rotXW = if tClamped < 1f then tClamped * 90f else 90f
    val rotYW = if tClamped < 1f then 0f else if tClamped < 2f then (tClamped - 1f) * 90f else 90f
    val rotZW = if tClamped < 2f then 0f else (tClamped - 2f) * 90f

    Scene(
      camera = Camera(
        position = (0f, 2f, 6f),
        lookAt   = (0f, 0f, 0f)
      ),
      objects = List(
        TesseractSponge(
          spongeType = SurfaceSubdividing,
          level      = level,
          material   = Some(Material.Glass),
          size       = 2.0f,
          projection = Some(Projection4DSpec(
            eyeW    = EyeW,
            screenW = ScreenW,
            rotXW   = rotXW,
            rotYW   = rotYW,
            rotZW   = rotZW
          ))
        )
      ),
      lights = List(
        Directional(direction = (1f, -1f, -1f), intensity = 1.5f),
        Directional(direction = (-1f, -0.5f, 1f), intensity = 0.4f)
      ),
      envMap      = Some("cliffside_2k.hdr"),
      toneMapping = ToneMapping.Reinhard(exposure = 1.2f)
    )


