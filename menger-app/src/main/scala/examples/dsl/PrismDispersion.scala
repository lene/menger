package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: Spectral dispersion through a glass sphere.
 *
 * White directional light passes through a dispersive glass sphere and splits
 * into spectral colors via wavelength-dependent refraction (Cauchy IOR model).
 * Rendered with hero-wavelength sampling — each pixel gets one wavelength per frame.
 *
 * Usage: --scene examples.dsl.PrismDispersion
 */
object PrismDispersion:
  val scene = Scene(
    camera = Camera(
      position = (0f, 1.5f, 5f),
      lookAt = (0f, 0f, 0f)
    ),
    lights = List(
      Directional(direction = (1f, 0f, 0.2f), intensity = 3.0f)
    ),
    objects = List(
      Sphere(Vec3(0f, 0f, 0f), Material.GlassDispersive, 1.0f)
    ),
    planes = List(
      Plane(Y at -2, color = Color.White),
      Plane(Z at  3, color = Color(0.1f, 0.1f, 0.1f))
    ),
    render = Some(RenderSettings(
      accumulation = 32,
      denoise = DenoiseMode.Final
    ))
  )
