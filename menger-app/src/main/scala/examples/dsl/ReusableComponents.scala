package examples.dsl

import scala.language.implicitConversions

import menger.dsl._
import examples.dsl.common.Materials._
import examples.dsl.common.Lighting._

/**
 * Example: Using reusable materials and lighting
 *
 * This demonstrates how to import and use the predefined materials
 * and lighting setups from the common package. This approach promotes
 * consistency across scenes and reduces code duplication.
 *
 * The scene uses:
 * - Custom materials from Materials object (TintedGlass, BrushedGold, etc.)
 * - Pre-configured lighting from Lighting object (ThreePointLighting)
 *
 * This pattern is especially useful when:
 * - Building multiple related scenes with consistent look
 * - Creating scene variants with different objects but same lighting
 * - Establishing a material library for a project
 *
 * Usage: --scene examples.dsl.ReusableComponents
 */
object ReusableComponents:
  val scene = Scene(
    camera = Camera(
      position = (0f, 2f, 6f),
      lookAt = (0f, 0f, 0f)
    ),
    objects = List(
      // Left: Tinted glass sphere
      Sphere(
        pos = (-2f, 0f, 0f),
        material = TintedGlass,
        size = 0.8f
      ),
      // Center left: Brushed gold sphere
      Sphere(
        pos = (-0.7f, 0f, 0f),
        material = BrushedGold,
        size = 0.8f
      ),
      // Center right: Rose gold sphere
      Sphere(
        pos = (0.7f, 0f, 0f),
        material = RoseGold,
        size = 0.8f
      ),
      // Right: Pearl sphere
      Sphere(
        pos = (2f, 0f, 0f),
        material = Pearl,
        size = 0.8f
      )
    ),
    // Use pre-configured three-point lighting
    lights = ThreePointLighting,
    plane = Some(Plane(Y at -1, color = "#505050"))
  )

  SceneRegistry.register("reusable-components", scene)
