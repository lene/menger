package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: Animated sphere orbiting the origin.
 *
 * The t parameter maps to an angle (radians). The sphere traces a circle
 * in the XZ plane at y=0. Use t from 0 to 2*pi for a full orbit.
 *
 * Usage:
 *   --scene examples.dsl.OrbitingSphere --t 0.5
 *   --scene examples.dsl.OrbitingSphere --frames 100 --start-t 0 --end-t 6.28 --save-name orbit_%04d.png
 */
object OrbitingSphere:
  def scene(t: Float): Scene =
    val x = 2f * math.cos(t).toFloat
    val z = 2f * math.sin(t).toFloat
    Scene(
      camera = Camera(position = (0f, 3f, 6f), lookAt = (0f, 0f, 0f)),
      objects = List(
        Sphere(pos = Vec3(x, 0f, z), material = Material.Chrome, size = 0.5f)
      ),
      lights = List(
        Directional(direction = (1f, -1f, -1f), intensity = 2.0f)
      ),
      planes = List(Plane(Y at -1.5, color = "#FFFFFF"))
    )
