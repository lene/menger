package examples.dsl

import scala.language.implicitConversions

import menger.Projection4DSpec
import menger.dsl._

/**
 * Example: animated 4D sponge in front of a 360-degree video background.
 *
 * Usage:
 *   --scene examples.dsl.EnvMapVideoSponge \
 *     --texture-dir menger-geometry/src/test/resources/ --t 0.5
 *   --scene examples.dsl.EnvMapVideoSponge \
 *     --texture-dir menger-geometry/src/test/resources/ \
 *     --frames 24 --start-t 0 --end-t 1 --save-name env_video_%04d.png
 */
object EnvMapVideoSponge:
  private val EyeW = 3.0f
  private val ScreenW = 1.5f
  private val MaxT = 1.0f
  private val BaseLevel = 0.5f
  private val LevelSweep = 0.75f
  private val SpongeSize = 2.4f
  private val XWRotationSweep = 120.0f
  private val YWRotationSweep = 45.0f
  private val EnvVideoFps = 2.0

  def scene(t: Float): Scene =
    val progress = math.max(0f, math.min(t, MaxT))
    val level = BaseLevel + progress * LevelSweep

    Scene(
      camera = Camera(
        position = (0f, 1.1f, 4.2f),
        lookAt   = (0f, 0f, 0f)
      ),
      objects = List(
        TesseractSponge(
          spongeType = SurfaceSubdividing,
          level      = level,
          material   = Some(Material.Gold),
          size       = SpongeSize,
          projection = Some(Projection4DSpec(
            eyeW    = EyeW,
            screenW = ScreenW,
            rotXW   = progress * XWRotationSweep,
            rotYW   = progress * YWRotationSweep
          ))
        )
      ),
      lights = List(
        Directional(direction = (1f, -1f, -1f), intensity = 1.2f)
      ),
      envMapVideo = Some(EnvMapVideo(
        "video/two-frame-equirect-rgba.mov",
        VideoPlayback(
          timeMapping = VideoTimeMapping.TProgress,
          repeat = VideoRepeat.Loop,
          fpsOverride = Some(EnvVideoFps)
        )
      )),
      ibl = Some(IBL(strength = 0.7f, samples = 1)),
      toneMapping = ToneMapping.Reinhard(exposure = 1.0f)
    )
