package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Two-spheres caustics validation scene — the physical twin of
 * `scripts/caustics-validation/scenes/two-spheres.pbrt`.
 *
 * Two glass spheres side by side, each casting its own caustic on the shared floor. Where
 * [[CausticsCanonical]] validates single-object caustics, this scene exercises the multi-object
 * photon-emission path (Sprint 33.7, P7). Geometry matches the pbrt twin verbatim:
 *   Spheres: centers (-1.3,0,0) and (1.3,0,0), radius 0.8, glass IOR 1.5
 *   Floor:   Y = -2, diffuse reflectance 0.8
 *   Light:   point (0,10,0), intensity 500 (== pbrt "rgb I" 500, W/sr)
 *   Camera:  (0,1.5,6) looking at origin
 *   Tone map: None (linear), so the PFM dump is directly comparable to pbrt's EXR
 *
 * Usage: --scene examples.dsl.TwoSpheres --save-name out.pfm
 */
object TwoSpheres:
  val scene: Scene = Scene(
    camera = Camera(
      position = (0f, 4f, 5f),
      lookAt = (0f, -1f, 0f),
      up = (0f, 1f, 0f)
    ),
    objects = List(
      Sphere(pos = (-1.3f, 0f, 0f), material = Material.Glass, size = 0.8f),
      Sphere(pos = (1.3f, 0f, 0f), material = Material.Glass, size = 0.8f)
    ),
    lights = List(
      Point(position = (0f, 10f, 0f), intensity = 500.0f)
    ),
    planes = List(
      Plane(Y at -2, color = Color(0.8f, 0.8f, 0.8f))
    ),
    background = Some(Color.Black),
    toneMapping = ToneMapping.None,
    // Auto gather radius (initialRadius = None): the two r=0.8 spheres are smaller than the
    // canonical unit sphere, so a fixed 1.0 radius (Caustics.HighQuality) over-smooths their
    // caustics into a single blur. Auto-derive from geometry keeps them focused (Sprint 33.9).
    caustics = Some(Caustics(photonsPerIteration = 500000, iterations = 20, alpha = 0.8f)),
    render = Some(RenderSettings(shadows = true))
  )

  SceneRegistry.register("two-spheres", scene)

/**
 * Caustics-OFF twin of [[TwoSpheres]] — identical geometry, lighting, camera and tone map,
 * caustics disabled. Subtracting its render from the caustics-on render isolates the caustic
 * contribution (Sprint 33 L3 "caustic-delta" metric, see docs/sprints/SPRINT33.md).
 *
 * Usage: --scene examples.dsl.TwoSpheresOff --save-name off.pfm
 */
object TwoSpheresOff:
  val scene: Scene = TwoSpheres.scene.copy(caustics = None)

  SceneRegistry.register("two-spheres-off", scene)
