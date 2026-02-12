package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: Material showcase scene
 *
 * This demonstrates the different material presets available in the DSL:
 * - Glass: Transparent refractive material
 * - Chrome: Metallic reflective material
 * - Gold: Metallic material with warm color
 *
 * The three spheres are arranged in a row to showcase their different
 * optical properties under the same lighting conditions.
 *
 * Usage: --scene examples.dsl.ThreeMaterials
 */
object ThreeMaterials:
  val scene = Scene(
    camera = Camera(
      position = (0f, 1f, 5f),
      lookAt = (0f, 0f, 0f)
    ),
    objects = List(
      // Left: Glass sphere - transparent and refractive
      Sphere(
        pos = (-2f, 0f, 0f),
        material = Material.Glass,
        size = 0.8f
      ),
      // Center: Chrome sphere - highly reflective
      Sphere(
        pos = (0f, 0f, 0f),
        material = Material.Chrome,
        size = 0.8f
      ),
      // Right: Gold sphere - warm metallic
      Sphere(
        pos = (2f, 0f, 0f),
        material = Material.Gold,
        size = 0.8f
      )
    ),
    lights = List(
      // Main light from upper right
      Directional(
        direction = (1f, -1f, -1f),
        intensity = 1.2f
      ),
      // Fill light from left for better material visibility
      Directional(
        direction = (-1f, -0.5f, -1f),
        intensity = 0.5f
      )
    ),
    plane = Some(Plane(Y at -1, color = "#606060"))  // Gray floor
  )

  SceneRegistry.register("three-materials", scene)
