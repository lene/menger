package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Canonical caustics validation scene — the exact physical twin of
 * `scripts/caustics-validation/scenes/canonical-caustics.pbrt`.
 *
 * Unlike `CausticsReference` (whose camera/light were fudged to work around the
 * pre-Sprint-33 physics bugs), this scene matches the pbrt reference verbatim so the
 * validation harness compares like with like:
 *   Sphere:  origin, radius 1.0, glass IOR 1.5
 *   Floor:   Y = -2, diffuse reflectance 0.8
 *   Light:   point (0,10,0), intensity 500 (== pbrt "rgb I" 500, W/sr)
 *   Camera:  (0,1,4) looking at origin
 *   Tone map: None (linear), so the PFM dump is directly comparable to pbrt's EXR
 *
 * Usage: --scene examples.dsl.CausticsCanonical --save-name out.pfm --tonemap none
 */
object CausticsCanonical:
  val scene: Scene = Scene(
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
    caustics = Some(Caustics(photonsPerIteration = 500000, iterations = 20, alpha = 0.8f)),
    render = Some(RenderSettings(shadows = true))
  )

  SceneRegistry.register("caustics-canonical", scene)

/**
 * Caustics-OFF twin of [[CausticsCanonical]] — identical geometry, lighting, camera and
 * tone map, with caustics disabled. Subtracting its render from the caustics-on render
 * isolates the caustic contribution (the shared direct + ambient lighting cancels), which is
 * the Sprint 33 L3 "caustic-delta" validation metric (see docs/sprints/SPRINT33.md).
 *
 * Usage: --scene examples.dsl.CausticsCanonicalOff --save-name off.pfm
 */
object CausticsCanonicalOff:
  val scene: Scene = CausticsCanonical.scene.copy(caustics = None)

  SceneRegistry.register("caustics-canonical-off", scene)
