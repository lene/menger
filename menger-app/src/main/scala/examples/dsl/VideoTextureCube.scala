package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: rectangular video texture decoded as a still initial frame.
 *
 * Usage:
 *   --scene examples.dsl.VideoTextureCube --texture-dir menger-geometry/src/test/resources/
 */
object VideoTextureCube:
  val scene = Scene(
    camera = Camera(
      // Sprint 36 F1: moved closer (was z=4) — measured framing 34.3% of frame, one of
      // the worst offenders in the Sprint 36 F1 framing survey.
      position = (0f, 0.6f, 2.8f),
      lookAt = (0f, 0f, 0f)
    ),
    objects = List(
      Cube(
        pos = (0f, 0f, 0f),
        size = 1.4f,
        color = Some(Color.White),
        videoTexture = Some(VideoTexture("video/two-frame-rgba.mov"))
      )
    ),
    lights = List(
      Directional(direction = (1f, -1f, -1f), intensity = 1.5f)
    )
  )

  SceneRegistry.register("video-texture-cube", scene)
