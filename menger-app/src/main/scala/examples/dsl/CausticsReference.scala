package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Caustics reference scene matching the PBRT v4 reference-scene.pbrt parameters.
 *
 * PBRT scene: Camera at (0,4,8), point light at (0,10,0) intensity 500,
 * glass sphere at origin (IOR 1.5), floor at Y=-2 (gray 0.8), black background.
 *
 * Our renderer uses an infinite plane (vs PBRT's finite 20x20 quad), so camera is
 * adjusted to (0,0.5,10) to show the horizon. A directional light (0,-1,0) is added
 * as the first light for the caustics photon emitter (uses light[0].direction as
 * photon travel direction due to convention mismatch).
 *
 * Usage: --scene examples.dsl.CausticsReference
 */
object CausticsReference:
  val scene = Scene(
    camera = Camera(
      position = (0f, 1.5f, 10f),
      lookAt = (0f, -0.5f, 0f),
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
      // Light 1 (index 0): Directional for caustics photon emitter.
      // Photon emitter uses lights[0].direction as photon travel direction.
      // Direction (0,-1,0) = photons travel downward through sphere onto floor.
      Directional(
        direction = (0f, -1f, 0f),
        intensity = 0.0f  // Zero intensity: no contribution to scene shading
      ),
      // Light 2: Point light matching PBRT reference position.
      // PBRT intensity 500 with point light; our renderer needs calibration.
      Point(
        position = (0f, 10f, 0f),
        intensity = 180.0f
      )
    ),
    planes = List(
      Plane(Y at -2, color = Color(0.8f, 0.8f, 0.8f))
    ),
    background = Some(Color.Black),
    caustics = None // Disabled: get base scene right first
  )

  SceneRegistry.register("caustics-reference", scene)
