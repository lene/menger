package examples.dsl

import menger.dsl.*

/**
 * Example: Menger Sponge showcase scene
 *
 * This demonstrates a more complex scene with:
 * - Multiple Menger sponges of different types
 * - Various materials (glass, gold, chrome, copper)
 * - Multiple lights with different colors
 * - High-quality caustics enabled
 *
 * The scene showcases the different sponge types:
 * - VolumeFilling: Standard Menger sponge
 * - SurfaceUnfolding: Surface-based sponge variation
 * - CubeSponge: Cube-based sponge pattern
 */
object MengerShowcase:
  val scene = Scene(
    camera = Camera(
      position = (8f, 6f, 10f),
      lookAt = (0f, 0f, 0f)
    ),
    objects = List(
      // Center: Chrome volume-filling sponge
      Sponge(
        spongeType = VolumeFilling,
        level = 2f,
        material = Material.Chrome,
        size = 2.0f
      ),
      // Left: Glass surface-unfolding sponge
      Sponge(
        pos = (-4f, 0f, 0f),
        spongeType = SurfaceUnfolding,
        level = 2.5f,
        material = Material.Glass,
        size = 1.8f
      ),
      // Right: Gold cube sponge
      Sponge(
        pos = (4f, 0f, 0f),
        spongeType = CubeSponge,
        level = 2f,
        material = Material.Gold,
        size = 1.8f
      ),
      // Back: Copper volume sponge
      Sponge(
        pos = (0f, 0f, -4f),
        spongeType = VolumeFilling,
        level = 2f,
        material = Material.Copper,
        size = 1.5f
      )
    ),
    lights = List(
      // Main key light from upper right
      Directional(
        direction = (1f, -1f, -1f),
        intensity = 1.2f,
        color = Color.White
      ),
      // Fill light from upper left with warm tone
      Point(
        position = (-5f, 5f, 5f),
        intensity = 1.5f,
        color = "#FFFFCC"  // Warm white
      ),
      // Back light with cool tone
      Point(
        position = (5f, 5f, -5f),
        intensity = 1.0f,
        color = "#CCFFFF"  // Cool white
      )
    ),
    caustics = Caustics.HighQuality
  )
