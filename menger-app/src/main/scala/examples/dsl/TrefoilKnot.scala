package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: Trefoil knot rendered via the OptiX built-in curves primitive.
 *
 * A trefoil knot is the simplest non-trivial knot, parameterised by:
 *   x = sin(t) + 2*sin(2t)
 *   y = cos(t) - 2*cos(2t)
 *   z = -sin(3t)
 *
 * Control points are sampled from this curve and passed directly to OptiX
 * as a round cubic B-spline, producing a smooth tube without triangle meshes.
 *
 * Usage: --scene trefoil-knot
 */
object TrefoilKnot:
  private val Segments = 128

  private val knotPoints: Seq[Vec3] =
    (0 until Segments).map { i =>
      val t = 2f * math.Pi.toFloat * i / Segments
      Vec3(
        math.sin(t).toFloat + 2f * math.sin(2f * t).toFloat,
        math.cos(t).toFloat - 2f * math.cos(2f * t).toFloat,
        -math.sin(3f * t).toFloat
      )
    }

  val scene = Scene(
    camera = Camera(
      position = (0f, 0f, 8f),
      lookAt = (0f, 0f, 0f)
    ),
    objects = List(
      Curve(
        points = knotPoints,
        radius = 0.12f,
        closed = true,
        material = Some(Material.Chrome)
      )
    ),
    lights = List(
      Directional(direction = (1f, -1f, -1f), intensity = 2.0f),
      Directional(direction = (-1f, -0.5f, 1f), intensity = 0.8f)
    ),
    planes = List(
      Plane(Y at -3.0, color = "#CCCCCC")
    )
  )

  SceneRegistry.register("trefoil-knot", scene)
