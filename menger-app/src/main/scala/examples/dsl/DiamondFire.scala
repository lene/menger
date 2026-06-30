package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: Diamond "fire" — spectral dispersion on a high-dispersion diamond.
 *
 * Uses diamond-dispersive preset (Abbe V_d ≈ 33, IOR 2.42) for strong spectral
 * splitting. Placed on a plane under a point light to show colored refractions.
 *
 * Usage: --scene examples.dsl.DiamondFire
 */
object DiamondFire:
  val scene = Scene(
    camera = Camera(
      position = (2f, 2f, 3f),
      lookAt = (0f, 0f, 0f)
    ),
    lights = List(
      Point(position = (5f, 4f, 3f), intensity = 8.0f)
    ),
    objects = List(
      Sphere(Vec3(0f, 0.3f, 0f), Material.DiamondDispersive, 0.8f)
    ),
    planes = List(
      Plane(Y at -2, color = Color.White)
    ),
    render = Some(RenderSettings(
      accumulation = 64,
      denoise = DenoiseMode.Final
    ))
  )
