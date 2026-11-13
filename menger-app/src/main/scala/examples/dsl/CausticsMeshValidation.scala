package examples.dsl

import scala.language.implicitConversions
import scala.math._

import menger.dsl._

/**
 * Caustics validation on *tessellated* geometry, sharing the exact canonical scene
 * (camera / light / floor / caustics) as `CausticsCanonical` so the resulting caustic can be
 * compared directly to the pbrt-v4 analytic-sphere reference. Only the sphere representation
 * changes: here a parametric-equation triangle mesh instead of the analytic sphere primitive.
 * A matching caustic proves menger's triangle-mesh photon path reproduces the analytic result.
 *
 * The canonical scene is written out below (`canonicalScene`) so this file is self-contained
 * (menger#149: as a corpus example it could not resolve `CausticsCanonical`, which lives in
 * another file, so a model imitating it produced a scene that did not compile). The test "keep
 * the CausticsMeshValidation canonical scene equal to CausticsCanonical"
 * (ExampleScenesSuite) holds it equal to `CausticsCanonical.scene`, which stays the pbrt twin.
 *
 * Usage: --scene examples.dsl.CausticsCanonicalParametric --save-name out.pfm
 */
object CausticsCanonicalParametric:
  /** Camera, light, floor and caustics of the pbrt-v4 canonical scene; its sphere is replaced below. */
  private val canonicalScene: Scene = Scene(
    camera = Camera(
      position = (0f, 1.5f, 6f),
      lookAt = (0f, 0f, 0f),
      up = (0f, 1f, 0f)
    ),
    objects = List(
      Sphere(pos = (0f, 0f, 0f), material = Material.Glass, size = 1.0f)
    ),
    lights = List(
      Point(position = (0f, 10f, 0f), intensity = 500.0f)
    ),
    planes = List(
      Plane(Y at -2, color = Color(0.8f, 0.8f, 0.8f))
    ),
    background = Some(Color.Black),
    toneMapping = ToneMapping.None,
    caustics = Some(Caustics(photonsPerIteration = 500000, iterations = 20, radiusReduction = 0.8f)),
    render = Some(RenderSettings(shadows = true))
  )

  private val TwoPi = 2f * Pi.toFloat
  // Unit sphere at the origin (radius 1), matching the canonical scene's analytic Sphere(size=1).
  private val unitSphere = ParametricSurface(
    f = (u, v) => Vec3(cos(u).toFloat * sin(v).toFloat, cos(v).toFloat, sin(u).toFloat * sin(v).toFloat),
    uRange = (0f, TwoPi), vRange = (0f, Pi.toFloat),
    uSteps = 64, vSteps = 32,
    closedU = true, closedV = false,
    material = Some(Material.Glass)
  )

  val scene: Scene = canonicalScene.copy(objects = List(unitSphere))
  SceneRegistry.register("caustics-canonical-parametric", scene)

/** Caustics-OFF twin of [[CausticsCanonicalParametric]] for the caustic-delta metric. */
object CausticsCanonicalParametricOff:
  val scene: Scene = CausticsCanonicalParametric.scene.copy(caustics = None)
  SceneRegistry.register("caustics-canonical-parametric-off", scene)
