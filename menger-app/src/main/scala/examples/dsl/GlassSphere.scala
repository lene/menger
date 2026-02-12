package examples.dsl

import scala.language.implicitConversions

import menger.dsl.*

/**
 * Example: Glass sphere with caustics
 *
 * This demonstrates rendering a glass sphere with caustics enabled.
 * Caustics are the light patterns created when light passes through
 * or reflects off a transparent or reflective surface.
 *
 * The scene features:
 * - A glass sphere using Material.Glass preset
 * - High-quality caustics rendering (photon mapping)
 * - White floor plane to show caustic light patterns
 * - Simple lighting to emphasize caustics effect
 *
 * Caustics require photon mapping, which traces light paths from
 * the light sources through glass objects to create realistic
 * light concentration patterns on diffuse surfaces.
 *
 * Usage: --scene examples.dsl.GlassSphere
 */
object GlassSphere:
  val scene = Scene(
    camera = Camera(
      position = (0f, 2f, 5f),
      lookAt = (0f, 0f, 0f)
    ),
    objects = List(
      Sphere(
        pos = (0f, 0f, 0f),
        material = Material.Glass,
        size = 1.0f
      )
    ),
    lights = List(
      // Strong directional light to create visible caustics
      Directional(
        direction = (1f, -1f, -1f),
        intensity = 2.0f
      )
    ),
    plane = Some(
      // White floor to clearly show caustic patterns
      Plane(Y at -1.5, color = "#FFFFFF")
    ),
    caustics = Some(Caustics.HighQuality)
  )

  SceneRegistry.register("glass-sphere", scene)
