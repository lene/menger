package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

object DenoiseIblDemo:
  val scene = Scene(
    camera = Camera(
      position = (0f, 0.6f, 3f),
      lookAt   = (0f, 0f, 0f)
    ),
    objects = List(
      Sphere(
        material = Some(Material.Matte),
        size     = 1.0f
      )
    ),
    lights      = List.empty,
    envMap      = Some("rogland_sunset_2k.hdr"),
    ibl         = Some(IBL(strength = 1.0f, samples = 1)),
    toneMapping = ToneMapping.Reinhard(exposure = 1.0f),
    render      = Some(RenderSettings(accumulation = 2))
  )

  SceneRegistry.register("denoise-ibl-demo", scene)
