package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: Animated sponge with varying fractal level.
 *
 * The t parameter controls the sponge's fractal recursion level,
 * creating a "pulsing" effect as the sponge grows and shrinks.
 * Use t from 0 to 3 to see levels 0 through 3.
 *
 * Usage:
 *   --scene examples.dsl.PulsingSponge --t 1.5
 *   --scene examples.dsl.PulsingSponge --frames 60 --start-t 0 --end-t 3 --save-name pulse_%04d.png
 */
object PulsingSponge:
  def scene(t: Float): Scene =
    val clampedLevel = math.max(0f, math.min(t, 3f))
    Scene(
      camera = Camera(position = (3f, 2f, 5f), lookAt = (0f, 0f, 0f)),
      objects = List(
        Sponge(
          pos = Vec3(0f, 0f, 0f),
          spongeType = VolumeFilling,
          level = clampedLevel,
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
