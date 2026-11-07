package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: a `Sun` key light over a glass cube on a round plinth (menger#158).
 *
 * `Sun` is a distant disk-shaped light given by the direction it travels and its apparent
 * diameter (the real sun is 0.53 degrees). It lights the scene like a `Directional` of the same
 * intensity, but its disk is a real area light: the shadow gets a penumbra and, being the first
 * light of the scene, the sun is the one that emits the caustic photons (only the first light
 * does). `HdriSun` (`menger.tools.HdriSun <file.hdr>`) prints the `Sun` that matches the sun of
 * an HDRI. No HDRI here, so the scene renders without fetched assets.
 *
 * Usage: --scene examples.dsl.SunKeyLight
 */
object SunKeyLight:
  val scene: Scene = Scene(
    camera = Camera(
      position = (0f, 2.6f, 4.2f),
      lookAt = (0f, -0.2f, 0f)
    ),
    objects = List(
      Cylinder(
        p0 = Some(Vec3(0f, -0.65f, 0f)),
        p1 = Some(Vec3(0f, -0.45f, 0f)),
        radius = Some(1.8f),
        material = Some(Material.matte(Color(0.85f, 0.82f, 0.76f)))
      ),
      Cube(pos = (0f, -0.05f, 0f), material = Some(Material.Glass), size = 0.8f)
    ),
    lights = List(
      // From about 42 degrees up, behind the cube and to the left, so the shadow and the caustic
      // fall in front of it.
      Sun(direction = (0.6f, -0.7f, 0.5f))
    ),
    caustics = Some(Caustics(photonsPerIteration = 200000, iterations = 10)),
    render = Some(RenderSettings(shadows = true))
  )

  SceneRegistry.register("sun-key-light", scene)
