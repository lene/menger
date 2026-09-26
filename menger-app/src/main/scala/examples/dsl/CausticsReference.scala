package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Caustics reference scene — a glass sphere on a diffuse floor lit by a single point
 * light, using the same camera as [[CausticsCanonical]] (the pbrt-validated setup) so the
 * caustic is clearly visible: camera at (0,1.5,6) looking at the origin, point light
 * (0,10,0), glass sphere radius 1 at origin, floor at Y=-2 (gray 0.8), black background.
 * (Pre-Sprint-33 this scene used a fudged camera/intensity to work around physics bugs that
 * are now fixed — see the caustic-visibility note in docs/caustics/CAUSTICS.md.)
 *
 * Unlike `CausticsCanonical`, this scene is viewed as a tone-mapped LDR PNG, not compared to
 * a linear pbrt .pfm -- and a single-frame headless render doesn't apply `Scene.toneMapping`
 * at all (only the animation path does, `WithAnimation.scala`). So the point light's
 * intensity is tuned down from pbrt's physically-matched 500 (`CausticsCanonical`'s value)
 * to 150, which keeps the caustic ring visible without blowing the 0.8-albedo floor out to
 * solid white (usability review 2026-09 inbox item).
 *
 * Usage: --scene examples.dsl.CausticsReference
 */
object CausticsReference:
  val scene = Scene(
    camera = Camera(
      position = (0f, 1.5f, 6f),
      lookAt = (0f, 0f, 0f),
      up = (0f, 1f, 0f)
    ),
    objects = List(
      Sphere(
        pos = (0f, 0f, 0f),
        material = Material.Glass,
        size = 1.0f
      )
    ),
    lights = List(
      Point(
        position = (0f, 10f, 0f),
        intensity = 150.0f
      )
    ),
    planes = List(
      Plane(Y at -2, color = Color(0.8f, 0.8f, 0.8f))
    ),
    background = Some(Color.Black),
    caustics = Some(Caustics.HighQuality)
  )

  SceneRegistry.register("caustics-reference", scene)

/** Same scene as CausticsReference, kept as a distinct name for the manual test menu's
  * primitive-vs-mesh pairing with ParametricSphereCaustics. Both now use Caustics.HighQuality
  * (bumped from Caustics.Default, which was too low-budget for the caustic ring to be
  * visible to the eye — see docs/caustics/CAUSTICS.md) so the comparison is meaningful.
  *
  * Usage: --scene examples.dsl.CausticsReferenceDefault
  */
object CausticsReferenceDefault:
  val scene: Scene = CausticsReference.scene.copy(caustics = Some(Caustics.HighQuality))
  SceneRegistry.register("caustics-reference-default", scene)
