package examples.dsl

import scala.language.implicitConversions

import menger.dsl.*

/**
 * Example: Minimal scene definition
 *
 * This demonstrates the absolute minimum DSL setup:
 * - A single sphere with default material
 * - Default camera position
 * - Single directional light
 *
 * This is the simplest possible scene to get started with the DSL.
 *
 * Usage: --scene examples.dsl.SimpleScene
 */
object SimpleScene:
  val scene = Scene(
    camera = Camera.Default,
    objects = List(
      Sphere(Material.Chrome)
    ),
    lights = List(
      Directional((1f, -1f, -1f))
    )
  )

  // Register for short-name access
  SceneRegistry.register("simple", scene)
