package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: a glass sphere hovering over a round plinth (menger#156).
 *
 * The plinth is a wide, flat `Cylinder` given by the centres of its end caps (`p0`, `p1`) and a
 * radius. A sun-like directional light shines through the sphere at an angle, so its shadow and
 * its caustic, the light the glass focuses into the middle of the shadow, fall on the plinth's top
 * cap. A Cylinder is shaded opaque, so it takes an opaque material. The shadow on the cap needs
 * an optix-jni with menger#156's shader fix; with optix-jni 0.5.0 only the caustic shows.
 *
 * Usage: --scene examples.dsl.Plinth
 */
object Plinth:
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
      Sphere(pos = (0f, 0.45f, 0f), material = Material.Glass, size = 0.55f)
    ),
    lights = List(
      // Travels down and toward the camera, so the shadow and the caustic land in front of the
      // sphere instead of under it.
      Directional(direction = (0.5f, -1f, 0.5f))
    ),
    caustics = Some(Caustics(photonsPerIteration = 200000, iterations = 10)),
    render = Some(RenderSettings(shadows = true))
  )

  SceneRegistry.register("plinth", scene)
