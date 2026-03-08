package menger.dsl

import com.typesafe.scalalogging.LazyLogging
import menger.cli.LightSpec
import menger.cli.PlaneConfig
import menger.common.{Color => CommonColor}
import menger.config.CameraConfig
import menger.config.SceneConfig
import menger.optix.CausticsConfig

/** Extracted conversion from DSL Scene to rendering configs.
  *
  * Reusable by both Main (single scene) and AnimatedOptiXEngine (per-frame scene).
  */
object SceneConverter extends LazyLogging:

  case class SceneConfigs(
    scene: SceneConfig,
    camera: CameraConfig,
    lights: List[LightSpec],
    caustics: CausticsConfig,
    background: Option[CommonColor] = None,
    planes: List[PlaneConfig] = List.empty
  )

  def convert(dslScene: Scene, fallbackCaustics: CausticsConfig): SceneConfigs =
    validateSceneMaterials(dslScene)
    val scene = dslScene.toSceneConfig
    val camera = dslScene.toCameraConfig
    val lights = dslScene.lights.map { light =>
      val commonLight = light.toCommonLight
      LightSpec.fromCommonLight(commonLight)
    }
    val caustics = dslScene.caustics.map(_.toCausticsConfig).getOrElse(fallbackCaustics)
    val background = dslScene.background.map(_.toCommonColor)
    val planes = dslScene.planes.map(_.toPlaneConfig)
    SceneConfigs(scene, camera, lights, caustics, background, planes)

  private def validateSceneMaterials(dslScene: Scene): Unit =
    dslScene.objects.foreach {
      case obj: Sphere         => obj.material.foreach(warnMaterial)
      case obj: Cube           => obj.material.foreach(warnMaterial)
      case obj: Sponge         => obj.material.foreach(warnMaterial)
      case obj: Tesseract      =>
        obj.material.foreach(warnMaterial)
        obj.edgeMaterial.foreach(warnMaterial)
      case obj: TesseractSponge =>
        obj.material.foreach(warnMaterial)
        obj.edgeMaterial.foreach(warnMaterial)
    }

  private def warnMaterial(material: Material): Unit =
    material.validate().foreach(w => logger.warn(s"[Material] $w"))
