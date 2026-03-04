package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: Multi-light setup demonstration
 *
 * This demonstrates complex lighting with:
 * - Multiple directional lights from different angles
 * - Multiple point lights with different colors
 * - Warm and cool light mixing (color temperature)
 * - Different light intensities for depth and atmosphere
 *
 * The scene uses a five-light setup:
 * 1. Key light (main directional)
 * 2. Fill light (softer directional)
 * 3. Rim light (directional from behind)
 * 4. Warm point light (amber overhead)
 * 5. Cool point light (blue accent)
 *
 * This creates a more cinematic and dimensional look compared
 * to simple single-light setups.
 *
 * Usage: --scene examples.dsl.ComplexLighting
 */
object ComplexLighting:
  val scene = Scene(
    camera = Camera(
      position = (3f, 2f, 5f),
      lookAt = (0f, 0f, 0f)
    ),
    objects = List(
      // Center: Chrome sphere to show reflections
      Sphere(
        pos = (0f, 0f, 0f),
        material = Material.Chrome,
        size = 1.0f
      ),
      // Left: Gold sphere
      Sphere(
        pos = (-2.5f, 0f, 0f),
        material = Material.Gold,
        size = 0.7f
      ),
      // Right: Copper sphere
      Sphere(
        pos = (2.5f, 0f, 0f),
        material = Material.Copper,
        size = 0.7f
      )
    ),
    lights = List(
      // Key light: Main directional from upper right
      Directional(
        direction = (1f, -1f, -1f),
        intensity = 1.5f,
        color = Color.White
      ),
      // Fill light: Softer directional from upper left
      Directional(
        direction = (-1f, -0.5f, -1f),
        intensity = 0.6f,
        color = Color.White
      ),
      // Rim light: Directional from behind to create edge highlights
      Directional(
        direction = (0f, 0.3f, 1f),
        intensity = 0.8f,
        color = Color.White
      ),
      // Warm point light: Overhead amber accent
      Point(
        position = (0f, 5f, 0f),
        intensity = 1.2f,
        color = "#FFCC88"  // Warm amber
      ),
      // Cool point light: Side blue accent
      Point(
        position = (5f, 2f, 2f),
        intensity = 0.8f,
        color = "#88CCFF"  // Cool blue
      )
    ),
    planes = List(
      // Checkered floor for depth perception
      Plane.checkered(Y at -1.5, ("#FFFFFF", "#888888"))
    )
  )

  SceneRegistry.register("complex-lighting", scene)
