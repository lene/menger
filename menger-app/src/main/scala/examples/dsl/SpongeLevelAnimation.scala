package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: Smooth fractional-level sweep of a recursive IAS sponge.
 *
 * t ∈ [0, 1] sweeps the fractal level from 1.0 to 4.0.  Fractional
 * levels produce a cross-fade between adjacent integer levels, so the
 * transition is visually continuous rather than a hard jump.
 *
 * Usage:
 *   --scene examples.dsl.SpongeLevelAnimation --t 0.5
 *   --scene examples.dsl.SpongeLevelAnimation --frames 60 --start-t 0 --end-t 1 \
 *     --save-name sponge_level_%04d.png
 */
object SpongeLevelAnimation:
  def scene(t: Float): Scene =
    val level = 1f + t * 3f
    Scene(
      camera = Camera(position = (3f, 2f, 5f), lookAt = (0f, 0f, 0f)),
      objects = List(
        Sponge(
          spongeType = RecursiveIAS,
          level = level,
          material = Material.Gold,
          size = 1.5f
        )
      ),
      lights = List(
        Directional(direction = (1f, -1f, -1f), intensity = 1.5f),
        Directional(direction = (-1f, -0.5f, 1f), intensity = 0.5f)
      ),
      planes = List(Plane(Y at -2, color = "#303030"))
    )
