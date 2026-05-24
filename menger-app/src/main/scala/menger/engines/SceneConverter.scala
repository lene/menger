package menger.engines

import com.typesafe.scalalogging.LazyLogging
import menger.ObjectRotation
import menger.ObjectSpec
import menger.common.FogConfig
import menger.common.Light
import menger.common.{Color => CommonColor}
import menger.config.CameraConfig
import menger.config.PlaneConfig
import menger.config.SceneConfig
import menger.dsl.Material
import menger.dsl.Scene
import menger.dsl.SceneNode
import menger.dsl.ToneMapping
import menger.dsl.Transform
import menger.optix.CausticsConfig
import menger.optix.RenderConfig

/** Extracted conversion from DSL Scene to rendering configs.
  *
  * Reusable by both Main (single scene) and AnimatedOptiXEngine (per-frame scene).
  */
object SceneConverter extends LazyLogging:

  case class SceneConfigs(
    scene: SceneConfig,
    camera: CameraConfig,
    lights: List[Light],
    caustics: CausticsConfig,
    background: Option[CommonColor] = None,
    planes: List[PlaneConfig] = List.empty,
    render: Option[RenderConfig] = None,
    fog: Option[FogConfig] = None,
    envMap: Option[String] = None,
    toneMappingOperator: Int = 0,
    toneMappingExposure: Float = 1.0f
  )

  def convert(dslScene: Scene, fallbackCaustics: CausticsConfig): SceneConfigs =
    validateSceneMaterials(dslScene)
    val objectSpecs = dslScene.root match
      case Some(node) => flattenNode(node, Transform.Identity, None)
      case None       => dslScene.objects.map(_.toObjectSpec)
    val scene      = SceneConfig.multiObject(objectSpecs)
    val camera     = CameraConfig(
      position = dslScene.camera.position.toGdxVector3,
      lookAt   = dslScene.camera.lookAt.toGdxVector3,
      up       = dslScene.camera.up.toGdxVector3
    )
    val lights     = dslScene.lights.map(_.toCommonLight)
    val caustics   = dslScene.caustics.map(_.toCausticsConfig).getOrElse(fallbackCaustics)
    val background = dslScene.background.map(_.toCommonColor)
    val planes     = dslScene.planes.map(p => PlaneConfig(p.toPlaneSpec, Some(p.toPlaneColorSpec), p.material))
    val render     = dslScene.render.map(_.toRenderConfig)
    val fog        = dslScene.fog.map(f => FogConfig(f.density, f.color.toCommonColor))
    val envMap     = dslScene.envMap
    val (tmOp, tmExp) = toToneMappingParams(dslScene.toneMapping)
    SceneConfigs(scene, camera, lights, caustics, background, planes, render, fog, envMap, tmOp, tmExp)

  /** Flatten a SceneNode tree to a list of ObjectSpecs.
    *
    * Transform accumulation:
    *   child world transform = accumulate(parentWorld, node.transform)
    *   Applied to each leaf: position shifted by world translation + world scale,
    *   size multiplied by world scale, rotations added.
    *
    * Material inheritance:
    *   The nearest ancestor SceneNode.material applies to all descendants unless a
    *   descendant sets its own material. A geometry's own material takes highest precedence.
    *
    * @param node              Current node to process
    * @param parentWorldTransform World transform accumulated from the root to this node's parent
    * @param inheritedMaterial   Material resolved from ancestor nodes (nearest ancestor wins)
    * @return Flattened list of ObjectSpecs with world transforms and materials applied
    */
  def flattenNode(
    node: SceneNode,
    parentWorldTransform: Transform,
    inheritedMaterial: Option[Material]
  ): List[ObjectSpec] =
    val worldTransform    = Transform.accumulate(parentWorldTransform, node.transform)
    val effectiveMaterial = node.material.orElse(inheritedMaterial)
    val leafSpecs = node.geometry.toList.map { obj =>
      applyWorldTransform(applyInheritedMaterial(obj.toObjectSpec, effectiveMaterial), worldTransform)
    }
    val childSpecs = node.children.flatMap(child =>
      flattenNode(child, worldTransform, effectiveMaterial)
    )
    leafSpecs ++ childSpecs

  private def applyWorldTransform(spec: ObjectSpec, world: Transform): ObjectSpec =
    spec.copy(
      x    = world.translation.x + world.scale * spec.x,
      y    = world.translation.y + world.scale * spec.y,
      z    = world.translation.z + world.scale * spec.z,
      size = world.scale * spec.size,
      rotation = ObjectRotation(
        world.rotation.x + spec.rotX,
        world.rotation.y + spec.rotY,
        world.rotation.z + spec.rotZ
      )
    )

  private def applyInheritedMaterial(spec: ObjectSpec, inherited: Option[Material]): ObjectSpec =
    inherited match
      case None           => spec
      case Some(material) =>
        if spec.material.isDefined then spec
        else spec.copy(material = Some(material.toOptixMaterial))

  private def validateSceneMaterials(dslScene: Scene): Unit =
    dslScene.objects.foreach(_.materialsToValidate.foreach(warnMaterial))

  private def warnMaterial(material: Material): Unit =
    material.validate().foreach(w => logger.warn(s"[Material] $w"))

  private def toToneMappingParams(tm: ToneMapping): (Int, Float) = tm match
    case ToneMapping.None          => (0, 1.0f)
    case ToneMapping.Reinhard(exp) => (1, exp)
    case ToneMapping.ACES(exp)     => (2, exp)
