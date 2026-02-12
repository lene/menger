package examples.dsl

import scala.language.implicitConversions

import menger.dsl.*

/**
 * Example: Caustics rendering demonstration
 *
 * This demonstrates photon-mapped caustics with:
 * - Glass sphere creating caustic patterns
 * - Floor plane to display caustic light patterns
 * - High-quality caustics configuration
 * - Elevated camera to show caustic patterns on floor
 *
 * Caustics are light patterns created when light refracts through
 * transparent objects. This scene shows the characteristic bright
 * patterns that appear on surfaces beneath glass objects.
 *
 * Note: Caustics rendering is computationally expensive and may
 * take longer to render than standard scenes.
 *
 * Usage: --scene examples.dsl.CausticsDemo
 */
object CausticsDemo:
  val scene = Scene(
    camera = Camera(
      position = (2f, 3f, 4f),
      lookAt = (0f, -0.5f, 0f)
    ),
    objects = List(
      // Glass sphere positioned to create caustics on floor
      Sphere(
        pos = (0f, 0.5f, 0f),
        material = Material.Glass,
        size = 1.0f
      )
    ),
    lights = List(
      // Strong directional light to create clear caustics
      Directional(
        direction = (1f, -1f, -1f),
        intensity = 2.0f
      )
    ),
    plane = Some(
      // White floor to show caustic patterns clearly
      Plane(Y at -1, color = Color.White)
    ),
    caustics = Some(Caustics.HighQuality)
  )

  SceneRegistry.register("caustics-demo", scene)
