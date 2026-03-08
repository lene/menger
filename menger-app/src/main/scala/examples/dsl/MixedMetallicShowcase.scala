package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/** Example: Mixed-Metallic showcase
  *
  * Demonstrates the metallic parameter across a continuous range from 0.0 (fully dielectric)
  * to 1.0 (fully metallic). Five spheres are arranged in a row, each with a different metallic
  * value, but otherwise identical material parameters:
  *
  *   - color = #AAAAAA (silver-gray)
  *   - roughness = 0.3 (moderate gloss)
  *   - IOR = 1.0
  *
  * The leftmost sphere is purely dielectric (no metallic response); the rightmost is fully
  * metallic. The intermediate spheres show the gradual transition in shading character:
  * a dielectric reflects light from the surface only, while a metal absorbs and re-emits via
  * conductor Fresnel, tinting reflections with the base color.
  *
  * Usage: --scene examples.dsl.MixedMetallicShowcase
  */
object MixedMetallicShowcase:
  private val silverGray = Color("#AAAAAA")

  private def metallicSphere(x: Float, metallicValue: Float): Sphere =
    Sphere(
      pos = (x, 0f, 0f),
      material = Material(
        color = silverGray,
        roughness = 0.3f,
        ior = 1.0f,
        metallic = metallicValue
      ),
      size = 0.8f
    )

  val scene = Scene(
    camera = Camera(
      position = (0f, 1.5f, 6f),
      lookAt = (0f, 0f, 0f)
    ),
    objects = List(
      metallicSphere(-2f, 0.00f),  // fully dielectric
      metallicSphere(-1f, 0.25f),
      metallicSphere( 0f, 0.50f),
      metallicSphere( 1f, 0.75f),
      metallicSphere( 2f, 1.00f)   // fully metallic
    ),
    lights = List(
      Directional(
        direction = (1f, -1f, -1f),
        intensity = 1.5f
      )
    ),
    planes = List(Plane(Y at -1, color = "#606060"))
  )

  SceneRegistry.register("mixed-metallic", scene)
