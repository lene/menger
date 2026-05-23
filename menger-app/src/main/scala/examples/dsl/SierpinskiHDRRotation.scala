package examples.dsl

import scala.language.implicitConversions

import menger.Projection4DSpec
import menger.dsl._

/**
 * Animated 4D Sierpinski pentachoron: film body + copper edges, HDR background.
 *
 * Animation (t: 0 → 1, loops naturally):
 *   - 3D Y-axis rotation: one full turn per t unit
 *   - 4D XW rotation: 0°→90° over t ∈ [0,1]
 *   - Fractional level fixed at 2.5
 *
 * Film material produces iridescent interference fringes that shift as the
 * 4D projection changes. Copper edges frame the facets.
 * Cliffside HDR (high-contrast midday sun) provides bright reflections.
 *
 * Usage:
 *   --scene examples.dsl.SierpinskiHDRRotation --texture-dir menger-app/src/test/resources/ --t 0.5
 *   --scene examples.dsl.SierpinskiHDRRotation --texture-dir menger-app/src/test/resources/ \
 *     --frames 60 --start-t 0 --end-t 1 --save-name sierpinski_hdr_%04d.png
 */
object SierpinskiHDRRotation:
  private val Level   = 2.5f
  private val EyeW    = 3.0f
  private val ScreenW = 1.5f
  private val TwoPi   = (2.0 * math.Pi).toFloat

  def scene(t: Float): Scene =
    val rotY3D = t * TwoPi
    val rotXW  = t * 90f

    Scene(
      camera = Camera(
        position = (0f, 2f, 5f),
        lookAt   = (0f, 0f, 0f)
      ),
      objects = List(
        Sierpinski4D(
          level       = Level,
          material    = Some(Material.Film),
          size        = 2.0f,
          edgeMaterial = Some(Material.Copper),
          edgeRadius  = Some(0.04f),
          projection  = Some(Projection4DSpec(
            eyeW    = EyeW,
            screenW = ScreenW,
            rotXW   = rotXW
          )),
          rotation = Vec3(0f, rotY3D, 0f)
        )
      ),
      lights = List(
        Directional(direction = (1f, -1f, -1f), intensity = 1.8f),
        Directional(direction = (-1f, -0.3f, 0.5f), intensity = 0.5f)
      ),
      envMap      = Some("cliffside_2k.hdr"),
      toneMapping = ToneMapping.Reinhard(exposure = 1.2f)
    )


