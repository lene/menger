package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: DSL-controlled render quality settings
 *
 * Demonstrates that render quality (shadows, antialiasing) can be set
 * in the scene definition itself — no CLI flags needed. CLI flags can
 * still override DSL values when explicitly supplied.
 *
 * Usage: --scene examples.dsl.RenderSettingsDemo
 */
object RenderSettingsDemo:
  val scene = Scene(
    camera = Camera(Vec3(0f, 1f, 4f), Vec3.Zero),
    objects = List(
      Sphere(Vec3(-1f, 0f, 0f), Material.Chrome, 0.8f),
      Sphere(Vec3(1f, 0f, 0f), Material.Glass, 0.8f)
    ),
    lights = List(
      Directional(Vec3(1f, -1f, -1f)),
      Point(Vec3(0f, 3f, 2f), 1.5f)
    ),
    planes = List(Plane(Y at -1f, Color(0.6f, 0.6f, 0.6f))),
    render = Some(RenderSettings(
      shadows = true,
      antialiasing = true,
      aaMaxDepth = 3,
      aaThreshold = 0.05f
    ))
  )

  SceneRegistry.register("render-settings-demo", scene)
