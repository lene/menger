package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: Menger Sponge showcase
 *
 * This demonstrates the Menger Sponge fractal, the signature
 * feature of the Menger ray tracer. The Menger Sponge is a
 * three-dimensional fractal created by recursively subdividing
 * a cube and removing smaller cubes.
 *
 * The scene features:
 * - A level 2 VolumeFilling Menger Sponge
 * - Gold material to highlight the fractal's intricate geometry
 * - Three-point lighting to show depth and detail
 * - Dark floor plane for contrast
 *
 * Level interpretation:
 * - level = 0: Single cube
 * - level = 1: First subdivision (20 cubes remain)
 * - level = 2: Second subdivision (400 cubes remain)
 * - level = 2.5: Interpolated state between level 2 and 3
 *
 * The VolumeFilling type creates the classic Menger Sponge
 * by removing cube volumes at each iteration. This creates
 * the characteristic fractal structure with infinite surface
 * area but zero volume as the level approaches infinity.
 *
 * Usage: --scene examples.dsl.MengerShowcase
 */
object MengerShowcase:
  val scene = Scene(
    camera = Camera(
      position = (4f, 3f, 6f),
      lookAt = (0f, 0f, 0f)
    ),
    objects = List(
      Sponge(
        pos = (0f, 0f, 0f),
        spongeType = VolumeFilling,
        level = 2f,
        material = Material.Gold,
        size = 2.5f
      )
    ),
    lights = List(
      // Key light from upper right
      Directional(
        direction = (1f, -1f, -1f),
        intensity = 1.5f
      ),
      // Fill light from left to reduce shadows
      Directional(
        direction = (-1f, -0.5f, -1f),
        intensity = 0.5f
      ),
      // Rim light from behind for edge definition
      Directional(
        direction = (0f, 0.5f, 1f),
        intensity = 0.8f
      )
    ),
    plane = Some(Plane(Y at -3, color = "#404040"))
  )

  SceneRegistry.register("menger-showcase", scene)
