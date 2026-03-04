package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: Thin-film interference showcase (soap bubbles / oil slicks)
 *
 * Demonstrates physically-based thin-film interference using the Airy formula.
 * Each sphere has a different film thickness, producing different iridescent colors:
 *
 *   - 300nm: constructive interference for violet/blue
 *   - 500nm: constructive interference for green (default Material.Film)
 *   - 700nm: constructive interference for red/orange
 *
 * Physics: The film's refractive index (IOR 1.33, soap film) causes wavelength-dependent
 * phase shifts. The Airy reflectance R(λ) = 2r²(1-cos δ) / (1 + r⁴ - 2r² cos δ)
 * produces peaks at different visible wavelengths for each thickness. CIE 1931 XYZ
 * color matching functions convert the spectral reflectance to RGB.
 *
 * Usage: --scene examples.dsl.FilmSphere
 *        --scene film-sphere
 */
object FilmSphere:
  val scene = Scene(
    camera = Camera(
      position = (0f, 1.5f, 6f),
      lookAt = (0f, 0f, 0f)
    ),
    objects = List(
      // Left: 300nm film — violet/blue tint
      Sphere(
        pos = (-2.5f, 0f, 0f),
        material = Material.Film.copy(filmThickness = 300f),
        size = 0.9f
      ),
      // Center: 500nm film — green tint (default Film preset)
      Sphere(
        pos = (0f, 0f, 0f),
        material = Material.Film,
        size = 0.9f
      ),
      // Right: 700nm film — red/orange tint
      Sphere(
        pos = (2.5f, 0f, 0f),
        material = Material.Film.copy(filmThickness = 700f),
        size = 0.9f
      )
    ),
    lights = List(
      // Main light from upper-right-front: creates glancing-angle iridescence on sphere edges
      Directional(
        direction = (1f, -1f, -1f),
        intensity = 1.5f
      ),
      // Soft fill from the left to illuminate the shadowed side
      Directional(
        direction = (-1f, -0.3f, -0.5f),
        intensity = 0.4f
      )
    ),
    // Dark floor to contrast the iridescent colors
    planes = List(Plane(Y at -1.2, color = "#101010"))
  )

  SceneRegistry.register("film-sphere", scene)
