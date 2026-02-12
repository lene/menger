package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: Custom material creation
 *
 * This demonstrates various ways to create custom materials:
 * - Using Material() constructor with explicit parameters
 * - Using .copy() to modify existing material presets
 * - Using factory methods for common material types
 * - Custom colors with hex strings
 *
 * This scene shows five spheres with different custom materials:
 * 1. Brushed gold (modified preset)
 * 2. Tinted glass (custom glass with color)
 * 3. Custom plastic (factory method)
 * 4. Custom matte (factory method)
 * 5. Custom metal (factory method)
 *
 * Usage: --scene examples.dsl.CustomMaterials
 */
object CustomMaterials:
  val scene = Scene(
    camera = Camera(
      position = (0f, 2f, 8f),
      lookAt = (0f, 0f, 0f)
    ),
    objects = List(
      // Brushed gold: modify Gold preset to be less reflective
      Sphere(
        pos = (-4f, 0f, 0f),
        material = Material.Gold.copy(roughness = 0.4f),
        size = 0.8f
      ),
      // Tinted glass: custom glass with blue tint
      Sphere(
        pos = (-2f, 0f, 0f),
        material = Material.Glass.copy(
          color = Color(0.9f, 0.95f, 1.0f, 0.02f)  // Slight blue tint
        ),
        size = 0.8f
      ),
      // Custom plastic: cyan colored
      Sphere(
        pos = (0f, 0f, 0f),
        material = Material.plastic(Color("#00FFFF")),
        size = 0.8f
      ),
      // Custom matte: red matte surface
      Sphere(
        pos = (2f, 0f, 0f),
        material = Material.matte(Color.Red),
        size = 0.8f
      ),
      // Custom metal: purple metallic
      Sphere(
        pos = (4f, 0f, 0f),
        material = Material.metal(Color("#8000FF")),
        size = 0.8f
      )
    ),
    lights = List(
      Directional(
        direction = (1f, -1f, -1f),
        intensity = 1.2f
      ),
      // Soft fill light to show material properties
      Directional(
        direction = (-0.5f, -0.3f, 1f),
        intensity = 0.3f
      )
    ),
    plane = Some(Plane(Y at -1, color = "#404040"))
  )

  SceneRegistry.register("custom-materials", scene)
