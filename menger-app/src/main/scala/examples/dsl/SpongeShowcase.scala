package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: Menger Sponge types demonstration
 *
 * This demonstrates the three different sponge generation algorithms:
 * - VolumeFilling: Standard Menger sponge (removes cube volumes)
 * - SurfaceUnfolding: Surface-based sponge variation
 * - CubeSponge: Cube-based sponge pattern
 *
 * Each sponge is shown with a different material to highlight their
 * distinct geometric properties. The scene uses fractional levels
 * to show intermediate subdivision states.
 *
 * Level interpretation:
 * - level = 0: Single cube
 * - level = 1: First subdivision (20 cubes for VolumeFilling)
 * - level = 2: Second subdivision (400 cubes for VolumeFilling)
 * - level = 1.5: Interpolated between level 1 and 2
 *
 * Usage: --scene examples.dsl.SpongeShowcase
 */
object SpongeShowcase:
  val scene = Scene(
    camera = Camera(
      position = (5f, 3f, 7f),
      lookAt = (0f, 0f, 0f)
    ),
    objects = List(
      // Left: VolumeFilling sponge with Chrome
      Sponge(
        pos = (-3f, 0f, 0f),
        spongeType = VolumeFilling,
        level = 2f,
        material = Material.Chrome,
        size = 1.8f
      ),
      // Center: SurfaceUnfolding sponge with Glass
      Sponge(
        pos = (0f, 0f, 0f),
        spongeType = SurfaceUnfolding,
        level = 2f,
        material = Material.Glass,
        size = 1.8f
      ),
      // Right: CubeSponge with Gold
      Sponge(
        pos = (3f, 0f, 0f),
        spongeType = CubeSponge,
        level = 2f,
        material = Material.Gold,
        size = 1.8f
      )
    ),
    lights = List(
      // Key light from upper right
      Directional(
        direction = (1f, -1f, -1f),
        intensity = 1.5f
      ),
      // Fill light from left
      Directional(
        direction = (-1f, -0.5f, -1f),
        intensity = 0.6f
      )
    ),
    planes = List(Plane(Y at -2, color = "#303030"))
  )

  SceneRegistry.register("sponge-showcase", scene)
