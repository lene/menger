package examples.dsl

import scala.language.implicitConversions
import scala.math._

import menger.dsl._

/** Parametric sphere — compare to built-in Sphere for visual validation. */
object ParametricSphere:
  private val TwoPi = 2f * Pi.toFloat

  val scene: Scene = Scene(
    objects = List(ParametricSurface(
      f = (u, v) => Vec3(cos(u).toFloat * sin(v).toFloat, cos(v).toFloat, sin(u).toFloat * sin(v).toFloat),
      uRange = (0f, TwoPi), vRange = (0f, Pi.toFloat),
      closedU = true, closedV = false,
      material = Some(Material.Glass)
    )),
    lights = List(Directional(
      direction = (1f, -1f, -1f),
      intensity = 2.0f
    )),
    planes = List(Plane(Y at -1.5, color = "#FFFFFF"))
  )
  SceneRegistry.register("parametric-sphere", scene)

/** Parametric torus — closed in both u and v, glass material. */
object ParametricTorus:
  private val TwoPi = 2f * Pi.toFloat
  private val R = 1.0f
  private val r = 0.4f

  val scene: Scene = Scene(
    objects = List(ParametricSurface(
      f = (u, v) => Vec3(
        (R + r * cos(v).toFloat) * cos(u).toFloat,
        r * sin(v).toFloat,
        (R + r * cos(v).toFloat) * sin(u).toFloat
      ),
      uRange = (0f, TwoPi), vRange = (0f, TwoPi),
      closedU = true, closedV = true,
      material = Some(Material.Glass)
    )),
    lights = List(Directional(
      direction = (1f, -1f, -1f),
      intensity = 2.0f
    )),
    planes = List(Plane(Y at -1.5, color = "#FFFFFF"))
  )
  SceneRegistry.register("parametric-torus", scene)

/** Parametric wavy sheet — open surface with IOR. */
object ParametricWavySheet:
  val scene: Scene = Scene(
    objects = List(ParametricSurface(
      f = (u, v) => Vec3(u, 0.3f * sin(u * 2).toFloat * cos(v * 2).toFloat, v),
      uRange = (-2f, 2f), vRange = (-2f, 2f),
      uSteps = 64, vSteps = 64,
      closedU = false, closedV = false,
      ior = 1.5f
    )),
    lights = List(Directional(
      direction = (1f, -1f, -1f),
      intensity = 2.0f
    )),
    planes = List(Plane(Y at -1.5, color = "#FFFFFF"))
  )
  SceneRegistry.register("parametric-wavy-sheet", scene)

/** Parametric Moebius strip — film material. */
object ParametricMoebius:
  private val TwoPi = 2f * Pi.toFloat

  val scene: Scene = Scene(
    objects = List(ParametricSurface(
      f = (u, v) =>
        val halfU = u / 2f
        val w = v - 0.5f
        Vec3(
          (1f + w * cos(halfU).toFloat) * cos(u).toFloat,
          (1f + w * cos(halfU).toFloat) * sin(u).toFloat,
          w * sin(halfU).toFloat
        ),
      uRange = (0f, TwoPi), vRange = (0f, 1f),
      uSteps = 128, vSteps = 16,
      closedU = false, closedV = false,
      material = Some(Material.Film)
    )),
    lights = List(Directional(
      direction = (1f, -1f, -1f),
      intensity = 2.0f
    ))
  )
  SceneRegistry.register("parametric-moebius", scene)

/** Figure-8 Klein bottle with IOR. */
object ParametricKleinBottle:
  private val TwoPi = 2f * Pi.toFloat
  private val a = 2.0f

  val f: (Float, Float) => Vec3 = (u, v) =>
    val cosU = cos(u).toFloat; val sinU = sin(u).toFloat
    val cosHalfU = cos(u / 2f).toFloat; val sinHalfU = sin(u / 2f).toFloat
    val sinV = sin(v).toFloat; val sin2V = sin(2f * v).toFloat
    val r = a + cosHalfU * sinV - sinHalfU * sin2V
    Vec3(r * cosU, r * sinU, sinHalfU * sinV + cosHalfU * sin2V)

  val scene: Scene = Scene(
    objects = List(ParametricSurface(
      f = f,
      uRange = (0f, TwoPi), vRange = (0f, TwoPi),
      uSteps = 128, vSteps = 64,
      closedU = true, closedV = true,
      ior = 1.5f
    )),
    lights = List(Directional(
      direction = (1f, -1f, -1f),
      intensity = 2.0f
    ))
  )
  SceneRegistry.register("parametric-klein-bottle", scene)

/** Figure-8 Klein bottle with film material. */
object ParametricKleinBottleFilm:
  val scene: Scene = Scene(
    objects = List(ParametricSurface(
      f = ParametricKleinBottle.f,
      uRange = (0f, 2f * Pi.toFloat), vRange = (0f, 2f * Pi.toFloat),
      uSteps = 128, vSteps = 64,
      closedU = true, closedV = true,
      material = Some(Material.Film)
    )),
    lights = List(Directional(
      direction = (1f, -1f, -1f),
      intensity = 2.0f
    ))
  )
  SceneRegistry.register("parametric-klein-bottle-film", scene)
