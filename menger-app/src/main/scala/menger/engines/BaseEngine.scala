package menger.engines

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.Game
import com.typesafe.scalalogging.LazyLogging
import io.github.lene.optix.CameraState
import io.github.lene.optix.OptiXRendererWrapper
import io.github.lene.optix.SceneConfigurator
import menger.ObjectSpec
import menger.OptiXRenderResources
import menger.common.ObjectType
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.common.ValidationException
import menger.engines.scene.SceneBuilder

abstract class BaseEngine(maxInstances: Int)(using protected val profilingConfig: ProfilingConfig)
    extends Game with RenderEngine with LazyLogging:

  protected val rendererWrapper: OptiXRendererWrapper = OptiXRendererWrapper(maxInstances)
  protected val renderResources: OptiXRenderResources = OptiXRenderResources(0, 0)
  protected def sceneConfigurator: SceneConfigurator
  protected def cameraState: CameraState
  /** Concrete engines provide their effective `RenderConfig`; used by
    * scene-build to honour render-quality flags such as `--gpu-project-4d`. */
  protected def renderConfig: RenderConfig

  // Override in concrete engines that need auto-adjustment (e.g. InteractiveEngine)
  protected def computeEffectiveMaxInstances(builder: SceneBuilder, specs: List[ObjectSpec]): Int =
    maxInstances

  /** Compute the max-instances budget required to host `specs`, accounting for
    * mixed-scene splits (sphere / cube-sponge / other-mesh groups). Mirrors
    * the dispatch logic in `buildMixedSceneObjects` so the renderer can be
    * reinitialised at the right size before scene construction. */
  protected def requiredMaxInstancesFor(specs: List[ObjectSpec]): Int =
    if specs.isEmpty then maxInstances
    else
      val analyticalSpecs = specs.filter(s => ObjectType.isAnalyticalPrimitive(s.objectType))
      val cubeSpongeSpecs = specs.filter(_.objectType.toLowerCase == "cube-sponge")
      val otherMeshSpecs  = specs.filterNot(s =>
        ObjectType.isAnalyticalPrimitive(s.objectType) ||
        s.objectType.toLowerCase == "cube-sponge")
      val analyticalMax = if analyticalSpecs.nonEmpty then
        analyticalSpecs.groupBy(_.objectType.toLowerCase).values.map { group =>
          GeometryRegistry.builderFor(group, textureDir, renderConfig.gpuProject4D)
            .map(b => computeEffectiveMaxInstances(b, group)).getOrElse(0)
        }.maxOption.getOrElse(0)
      else 0
      val cubeSpongeMax = if cubeSpongeSpecs.nonEmpty then
        GeometryRegistry.builderFor(cubeSpongeSpecs, textureDir, renderConfig.gpuProject4D)
          .map(b => computeEffectiveMaxInstances(b, cubeSpongeSpecs)).getOrElse(0) else 0
      val otherMeshMax = if otherMeshSpecs.nonEmpty then
        GeometryRegistry.builderFor(otherMeshSpecs, textureDir, renderConfig.gpuProject4D)
          .map(b => computeEffectiveMaxInstances(b, otherMeshSpecs)).getOrElse(0) else 0
      Math.max(analyticalMax, Math.max(cubeSpongeMax, otherMeshMax))

  // Must be provided by concrete engine — where texture assets live
  protected def textureDir: String

  protected def buildSceneFromSpecs(
    specs: List[ObjectSpec],
    renderer: io.github.lene.optix.OptiXRenderer
  ): Try[Unit] =
    RenderModeSelector.classify(specs) match
      case SceneType.SimpleMixed(allSpecs, _) =>
        val analyticalSpecs = allSpecs.filter(s => ObjectType.isAnalyticalPrimitive(s.objectType))
        val meshSpecs = allSpecs.filterNot(s => ObjectType.isAnalyticalPrimitive(s.objectType))
        logger.info(
          s"Mixed scene: ${analyticalSpecs.size} analytical + ${meshSpecs.size} mesh objects"
        )
        Try(buildMixedSceneObjects(analyticalSpecs, meshSpecs, renderer))

      case SceneType.Unsupported(allSpecs) =>
        val objectTypes = allSpecs.map(_.objectType).distinct
        Failure(UnsupportedOperationException(
          "Cannot mix analytical primitives with multiple different triangle mesh types. " +
          s"Objects: ${objectTypes.mkString(", ")}. " +
          "Analytical primitives can be mixed with one mesh type at a time."
        ))

      case sceneType =>
        GeometryRegistry.builderFor(specs, textureDir, renderConfig.gpuProject4D) match
          case Some(builder) =>
            val effectiveMaxInstances = computeEffectiveMaxInstances(builder, specs)
            builder.validate(specs, effectiveMaxInstances) match
              case Left(error) =>
                Failure(ValidationException(error, "objectSpecs", specs.map(_.objectType)))
              case Right(_) =>
                builder.buildScene(specs, renderer, effectiveMaxInstances)
          case None =>
            Failure(UnsupportedOperationException(s"No builder available for $sceneType"))

  protected def buildSceneFromConfigs(
    configs: SceneConverter.SceneConfigs,
    renderer: io.github.lene.optix.OptiXRenderer
  ): Try[Unit] =
    val specs = configs.scene.objectSpecs.getOrElse(List.empty)
    val sceneType = RenderModeSelector.classify(specs)
    sceneType match
      case SceneType.TriangleMeshes(_) =>
        GeometryRegistry.builderFor(specs, textureDir, renderConfig.gpuProject4D) match
          case Some(builder) => builder.buildScene(specs, renderer, maxInstances)
          case None          => Failure(UnsupportedOperationException(s"No builder for $sceneType"))
      case SceneType.SimpleMixed(allSpecs, _) =>
        Try {
          val analyticalSpecs = allSpecs.filter(s => ObjectType.isAnalyticalPrimitive(s.objectType))
          val nonAnalyticalSpecs  = allSpecs.filterNot(s =>
            ObjectType.isAnalyticalPrimitive(s.objectType))
          val cubeSpongeSpecs = nonAnalyticalSpecs.filter(_.objectType.toLowerCase == "cube-sponge")
          val otherMeshSpecs  = nonAnalyticalSpecs.filterNot(
            _.objectType.toLowerCase == "cube-sponge")
          if analyticalSpecs.nonEmpty then
            analyticalSpecs.groupBy(_.objectType.toLowerCase).foreach { (objType, group) =>
              GeometryRegistry.builderFor(group, textureDir, renderConfig.gpuProject4D)
                .map(_.buildScene(group, renderer, maxInstances).get)
                .getOrElse(sys.error(s"No builder for analytical primitive type: $objType"))
            }
          if cubeSpongeSpecs.nonEmpty then
            GeometryRegistry.builderFor(cubeSpongeSpecs, textureDir, renderConfig.gpuProject4D)
              .map(_.buildScene(cubeSpongeSpecs, renderer, maxInstances).get)
              .getOrElse(sys.error("No builder for cube-sponge specs"))
          if otherMeshSpecs.nonEmpty then
            GeometryRegistry.builderFor(otherMeshSpecs, textureDir, renderConfig.gpuProject4D) match
              case Some(builder) => builder.buildScene(otherMeshSpecs, renderer, maxInstances).get
              case None =>
                val types = otherMeshSpecs.map(_.objectType).distinct.mkString(", ")
                sys.error(s"No mesh builder found for types: $types")
        }
      case other =>
        GeometryRegistry.builderFor(specs, textureDir, renderConfig.gpuProject4D) match
          case Some(builder) => builder.buildScene(specs, renderer, maxInstances)
          case None          =>
            Failure(UnsupportedOperationException(s"Unsupported scene type: $other"))

  protected def rebuildGeometry(
    specs: List[ObjectSpec],
    renderer: io.github.lene.optix.OptiXRenderer
  ): Unit =
    renderer.clearAllInstances()
    RenderModeSelector.classify(specs) match
      case SceneType.SimpleMixed(allSpecs, _) =>
        val analyticalSpecs = allSpecs.filter(s => ObjectType.isAnalyticalPrimitive(s.objectType))
        val meshSpecs = allSpecs.filterNot(s => ObjectType.isAnalyticalPrimitive(s.objectType))
        logger.debug(
          s"Rebuilding mixed scene: ${analyticalSpecs.size} analytical + ${meshSpecs.size} mesh"
        )
        buildMixedSceneObjects(analyticalSpecs, meshSpecs, renderer)

      case SceneType.Unsupported(_) =>
        sys.error("Complex mixed scenes not supported for rebuilding")

      case sceneType =>
        GeometryRegistry.builderFor(specs, textureDir, renderConfig.gpuProject4D) match
          case Some(builder) =>
            builder.buildScene(specs, renderer, maxInstances).get
          case None =>
            logger.warn(s"Cannot rebuild scene type: $sceneType")
            sys.error(s"Scene type $sceneType not supported for rebuilding")

  private def buildMixedSceneObjects(
    analyticalSpecs: List[ObjectSpec],
    meshSpecs: List[ObjectSpec],
    renderer: io.github.lene.optix.OptiXRenderer
  ): Unit =
    // cube-sponge specs need CubeSpongeSceneBuilder (instance-explosion path);
    // other triangle-mesh types go through TriangleMeshSceneBuilder
    // (H-sponge-showcase-crash fix).
    val cubeSpongeSpecs = meshSpecs.filter(_.objectType.toLowerCase == "cube-sponge")
    val otherMeshSpecs  = meshSpecs.filterNot(_.objectType.toLowerCase == "cube-sponge")
    // Auto-adjust budget across groups: each builder may have a very different
    // instance footprint (cube-sponge expands by 20^level, mesh builders are
    // 1:1). Take the max so cube-sponge dominance lifts the limit when needed.
    val analyticalMax = if analyticalSpecs.nonEmpty then
      analyticalSpecs.groupBy(_.objectType.toLowerCase).values.map { group =>
        GeometryRegistry.builderFor(group, textureDir, renderConfig.gpuProject4D)
          .map(b => computeEffectiveMaxInstances(b, group)).getOrElse(0)
      }.maxOption.getOrElse(0)
    else 0
    val cubeSpongeMax = if cubeSpongeSpecs.nonEmpty then
      GeometryRegistry.builderFor(cubeSpongeSpecs, textureDir, renderConfig.gpuProject4D)
        .map(b => computeEffectiveMaxInstances(b, cubeSpongeSpecs)).getOrElse(0) else 0
    val otherMeshMax = if otherMeshSpecs.nonEmpty then
      GeometryRegistry.builderFor(otherMeshSpecs, textureDir, renderConfig.gpuProject4D)
        .map(b => computeEffectiveMaxInstances(b, otherMeshSpecs)).getOrElse(0) else 0
    val effectiveMaxInstances = Math.max(analyticalMax, Math.max(cubeSpongeMax, otherMeshMax))
    if analyticalSpecs.nonEmpty then
      analyticalSpecs.groupBy(_.objectType.toLowerCase).foreach { (objType, group) =>
        GeometryRegistry.builderFor(group, textureDir, renderConfig.gpuProject4D)
          .map(_.buildScene(group, renderer, effectiveMaxInstances).get)
          .getOrElse(sys.error(s"No builder for analytical primitive type: $objType"))
      }
    if cubeSpongeSpecs.nonEmpty then
      GeometryRegistry.builderFor(cubeSpongeSpecs, textureDir, renderConfig.gpuProject4D)
        .map(_.buildScene(cubeSpongeSpecs, renderer, effectiveMaxInstances).get)
        .getOrElse(sys.error("No builder for cube-sponge specs"))
    if otherMeshSpecs.nonEmpty then
      GeometryRegistry.builderFor(otherMeshSpecs, textureDir, renderConfig.gpuProject4D)
        .map(_.buildScene(otherMeshSpecs, renderer, effectiveMaxInstances).get)
        .getOrElse {
          val types = otherMeshSpecs.map(_.objectType).distinct.mkString(", ")
          sys.error(s"No mesh builder found for types: $types")
        }

  // Default lifecycle — concrete engines override what they need
  override def create(): Unit = {}
  override def render(): Unit = {}
  override def resize(width: Int, height: Int): Unit = {}
  override def dispose(): Unit =
    renderResources.dispose()
    rendererWrapper.dispose()
  override def pause(): Unit  = {}
  override def resume(): Unit = {}
