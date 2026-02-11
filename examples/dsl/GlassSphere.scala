package examples.dsl

import menger.dsl.*

/**
 * Example: Simple glass sphere scene
 *
 * This demonstrates the basic DSL for defining a scene with:
 * - A single glass sphere at the origin
 * - A camera positioned to view the sphere
 * - A directional light
 *
 * Usage: --scene glass-sphere
 */
object GlassSphere:
  val scene = Scene(
    camera = Camera(
      position = (0f, 0f, 3f),
      lookAt = (0f, 0f, 0f)
    ),
    objects = List(
      Sphere(Material.Glass)
    ),
    lights = List(
      Directional((1f, -1f, -1f))
    )
  )

  // Register scene for CLI access
  SceneRegistry.register("glass-sphere", scene)
