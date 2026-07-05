package examples.dsl

import scala.language.implicitConversions
import scala.math._

import menger.dsl._

/**
 * Caustics validation on *tessellated* geometry, sharing the exact canonical scene
 * (camera / light / floor / caustics) as [[CausticsCanonical]] so the resulting caustic can be
 * compared directly to the pbrt-v4 analytic-sphere reference. Only the sphere representation
 * changes: here a parametric-equation triangle mesh instead of the analytic sphere primitive.
 * A matching caustic proves menger's triangle-mesh photon path reproduces the analytic result.
 *
 * Usage: --scene examples.dsl.CausticsCanonicalParametric --save-name out.pfm
 */
object CausticsCanonicalParametric:
  private val TwoPi = 2f * Pi.toFloat
  // Unit sphere at the origin (radius 1), matching CausticsCanonical's analytic Sphere(size=1).
  private val unitSphere = ParametricSurface(
    f = (u, v) => Vec3(cos(u).toFloat * sin(v).toFloat, cos(v).toFloat, sin(u).toFloat * sin(v).toFloat),
    uRange = (0f, TwoPi), vRange = (0f, Pi.toFloat),
    uSteps = 64, vSteps = 32,
    closedU = true, closedV = false,
    material = Some(Material.Glass)
  )

  val scene: Scene = CausticsCanonical.scene.copy(objects = List(unitSphere))
  SceneRegistry.register("caustics-canonical-parametric", scene)

/** Caustics-OFF twin of [[CausticsCanonicalParametric]] for the caustic-delta metric. */
object CausticsCanonicalParametricOff:
  val scene: Scene = CausticsCanonicalParametric.scene.copy(caustics = None)
  SceneRegistry.register("caustics-canonical-parametric-off", scene)
