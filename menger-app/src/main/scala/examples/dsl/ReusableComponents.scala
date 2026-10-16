package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: Using reusable materials and lighting
 *
 * This demonstrates defining materials and a lighting setup once, as values, and reusing them.
 * The same definitions live in the `examples.dsl.common` library (`Materials`, `Lighting`) for
 * scenes inside this repository; they are written out here so the scene is self-contained
 * (review 2026-10-14, menger#115: as a corpus example its `examples.dsl.common` imports were
 * rejected by the scene agent's allowlist, since they do not resolve for a standalone scene).
 *
 * The scene uses:
 * - Custom materials derived from presets (TintedGlass, BrushedGold, RoseGold, Pearl)
 * - A three-point lighting setup
 *
 * This pattern is especially useful when:
 * - Building multiple related scenes with consistent look
 * - Creating scene variants with different objects but same lighting
 * - Establishing a material library for a project
 *
 * Usage: --scene examples.dsl.ReusableComponents
 */
object ReusableComponents:
  /** Blue-tinted glass for underwater or ice effects */
  val TintedGlass = Material.Glass.copy(color = Color(0.85f, 0.9f, 1.0f, 0.02f))
  /** Brushed gold with higher roughness for matte finish */
  val BrushedGold = Material.Gold.copy(roughness = 0.4f)
  /** Rose gold alloy */
  val RoseGold = Material.metal(Color(0.85f, 0.5f, 0.5f)).copy(roughness = 0.1f)
  /** Pearl - subtle iridescent white */
  val Pearl = Material(
    color = Color(0.95f, 0.95f, 1.0f),
    roughness = 0.15f,
    metallic = 0.0f,
    specular = 0.8f
  )

  /** Classic three-point lighting: key from upper right, fill from the left, rim from behind */
  val ThreePointLighting = List(
    Directional(direction = (1f, -1f, -1f), intensity = 1.5f),
    Directional(direction = (-1f, -0.5f, -1f), intensity = 0.5f),
    Directional(direction = (0f, 0.5f, 1f), intensity = 0.8f)
  )

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
    lights = ThreePointLighting,
    planes = List(Plane(Y at -1, color = "#505050"))
  )

  SceneRegistry.register("reusable-components", scene)
