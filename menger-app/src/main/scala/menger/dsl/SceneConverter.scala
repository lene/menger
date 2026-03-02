package menger.dsl

import menger.cli.LightSpec
import menger.common.{Color => CommonColor}
import menger.config.CameraConfig
import menger.config.SceneConfig
import menger.optix.CausticsConfig

/** Extracted conversion from DSL Scene to rendering configs.
  *
  * Reusable by both Main (single scene) and AnimatedOptiXEngine (per-frame scene).
  */
object SceneConverter:

  case class SceneConfigs(
    scene: SceneConfig,
    camera: CameraConfig,
    lights: List[LightSpec],
    caustics: CausticsConfig,
    background: Option[CommonColor] = None
  )

  def convert(dslScene: Scene, fallbackCaustics: CausticsConfig): SceneConfigs =
    val scene = dslScene.toSceneConfig
    val camera = dslScene.toCameraConfig
    val lights = dslScene.lights.map { light =>
      val commonLight = light.toCommonLight
      LightSpec.fromCommonLight(commonLight)
    }
    val caustics = dslScene.caustics.map(_.toCausticsConfig).getOrElse(fallbackCaustics)
    val background = dslScene.background.map(_.toCommonColor)
    SceneConfigs(scene, camera, lights, caustics, background)
