package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Minimal IBL verification scene: a matte sphere with no explicit lights.
 *
 * With IBL enabled the sphere is lit by the HDR environment map (diffuse
 * illumination from the sky).  With IBL disabled (background-only env map)
 * the sphere is near-black, because there is no other light source.
 *
 * This contrast is the clearest visual check that IBL is actually working.
 *
 * Usage:
 *   --scene examples.dsl.IblSphereDemo --texture-dir menger-app/src/test/resources/
 *
 * To see the difference, compare with background-only mode:
 *   --objects type=sphere:material=matte \
 *     --env-map rogland_sunset_2k.hdr \
 *     --texture-dir menger-app/src/test/resources/
 * (sphere will be black — no light source without IBL)
 */
object IblSphereDemo:
  val scene = Scene(
    camera = Camera(
      position = (0f, 0.5f, 3f),
      lookAt   = (0f, 0f, 0f)
    ),
    objects = List(
      Sphere(
        material = Some(Material.Matte),
        size     = 1.0f
      )
    ),
    lights      = List.empty,          // deliberately no explicit lights
    envMap      = Some("rogland_sunset_2k.hdr"),
    ibl         = Some(IBL(strength = 1.0f, samples = 1)),
    toneMapping = ToneMapping.Reinhard(exposure = 1.0f)
    // accumulation intentionally omitted (defaults to 1) — keeps integration
    // test fast while still producing a deterministic reference image
  )

  SceneRegistry.register("ibl-sphere-demo", scene)
