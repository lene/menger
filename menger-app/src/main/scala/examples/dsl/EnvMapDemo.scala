package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Manual interactive test for HDR environment map DSL wiring (Task 22.1).
 *
 * What to verify:
 *   - Background shows equirectangular panorama instead of solid color
 *   - Sphere reflects/refracts against panorama (visible in glass material)
 *   - No crash on startup or scene load
 *
 * Prerequisites:
 *   - An HDR panorama file (Radiance .hdr format, equirectangular projection)
 *   - Place it in your texture directory or pass --texture-dir pointing at its parent
 *   - Test file: menger-app/src/test/resources/rogland_sunset_2k.hdr
 *
 * Usage:
 *   --scene examples.dsl.EnvMapDemo --texture-dir menger-app/src/test/resources/
 *
 * If the path is wrong you will see an error log line but no crash.
 */
object EnvMapDemo:
  val scene = Scene(
    camera = Camera(
      position = (0f, 1f, 4f),
      lookAt   = (0f, 0f, 0f)
    ),
    objects = List(
      Cube(
        pos      = (0f, 0f, 0f),
        material = Material.Glass,
        size     = 1.0f
      )
    ),
    lights = List(
      Directional(direction = (1f, -1f, -1f), intensity = 1.5f)
    ),
    envMap = Some("rogland_sunset_2k.hdr")
  )

  SceneRegistry.register("env-map-demo", scene)
