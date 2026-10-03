package menger.engines

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.Game
import com.typesafe.scalalogging.LazyLogging
import io.github.lene.optix.CameraState
import io.github.lene.optix.OptiXRenderer
import io.github.lene.optix.OptiXRendererWrapper
import io.github.lene.optix.SceneConfigurator
import menger.ObjectSpec
import menger.OptiXRenderResources
import menger.common.ObjectType
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.dsl.DenoiseMode
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
  protected def denoiseMode: DenoiseMode = DenoiseMode.Off
  protected def accumulationFrames: Int = 1

  protected def configureOutputMode(renderer: OptiXRenderer): Unit =
    renderer.setDenoisingEnabled(denoiseMode == DenoiseMode.Final)
    if accumulationFrames > 1 then renderer.setAccumulationFrames(accumulationFrames)

  // Override in concrete engines that need auto-adjustment (e.g. InteractiveEngine)
  protected def computeEffectiveMaxInstances(builder: SceneBuilder, specs: List[ObjectSpec]): Int =
    maxInstances

  /** Compute the max-instances budget required to host `specs`, accounting for
    * mixed-scene splits (`SceneGroups.buildOrder`). Mirrors the dispatch logic in
    * `buildMixedSceneObjects` so the renderer can be reinitialised at the right size before
    * scene construction. */
  protected def requiredMaxInstancesFor(specs: List[ObjectSpec]): Int =
    if specs.isEmpty then maxInstances
    else maxInstancesForGroups(SceneGroups.buildOrder(specs))

  /** Each builder may have a very different instance footprint (cube-sponge expands by
    * 20^level, edge rendering adds one cylinder per edge, mesh builders are 1:1); take the max
    * so the dominant group lifts the limit when needed. */
  private def maxInstancesForGroups(groups: List[List[ObjectSpec]]): Int =
    groups.map { group =>
      GeometryRegistry.builderFor(group, textureDir)
        .map(b => computeEffectiveMaxInstances(b, group)).getOrElse(0)
    }.maxOption.getOrElse(0)

  // Must be provided by concrete engine — where texture assets live
  protected def textureDir: String

  protected def buildSceneFromSpecs(
    specs: List[ObjectSpec],
    renderer: io.github.lene.optix.OptiXRenderer
  ): Try[Unit] =
    RenderModeSelector.classify(specs) match
      case _ if SceneGroups.hasMixedEdge4D(specs) =>
        Try(buildMixedSceneObjects(specs, renderer))

      case SceneType.SimpleMixed(allSpecs, _) =>
        val analyticalCount = allSpecs.count(s => ObjectType.isAnalyticalPrimitive(s.objectType))
        logger.info(
          s"Mixed scene: $analyticalCount analytical + ${allSpecs.size - analyticalCount} mesh objects"
        )
        Try(buildMixedSceneObjects(allSpecs, renderer))

      case SceneType.Unsupported(allSpecs) =>
        val objectTypes = allSpecs.map(_.objectType).distinct
        Failure(UnsupportedOperationException(
          "Cannot mix analytical primitives with multiple different triangle mesh types. " +
          s"Objects: ${objectTypes.mkString(", ")}. " +
          "Analytical primitives can be mixed with one mesh type at a time."
        ))

      case sceneType =>
        GeometryRegistry.builderFor(specs, textureDir) match
          case Some(builder) =>
            val effectiveMaxInstances = computeEffectiveMaxInstances(builder, specs)
            builder.validateAndBuild(specs, renderer, effectiveMaxInstances)
          case None =>
            Failure(UnsupportedOperationException(s"No builder available for $sceneType"))

  protected def buildSceneFromConfigs(
    configs: SceneConverter.SceneConfigs,
    renderer: io.github.lene.optix.OptiXRenderer
  ): Try[Unit] =
    val specs = configs.scene.objectSpecs.getOrElse(List.empty)
    val sceneType = RenderModeSelector.classify(specs)
    sceneType match
      case _ if SceneGroups.hasMixedEdge4D(specs) =>
        Try(buildMixedSceneObjects(specs, renderer))
      case SceneType.TriangleMeshes(_) =>
        GeometryRegistry.builderFor(specs, textureDir) match
          case Some(builder) =>
            val effectiveMaxInstances = computeEffectiveMaxInstances(builder, specs)
            builder.validateAndBuild(specs, renderer, effectiveMaxInstances)
          case None => Failure(UnsupportedOperationException(s"No builder for $sceneType"))
      case SceneType.SimpleMixed(allSpecs, _) =>
        Try(buildMixedSceneObjects(allSpecs, renderer))
      case other =>
        GeometryRegistry.builderFor(specs, textureDir) match
          case Some(builder) =>
            val effectiveMaxInstances = computeEffectiveMaxInstances(builder, specs)
            builder.validateAndBuild(specs, renderer, effectiveMaxInstances)
          case None =>
            Failure(UnsupportedOperationException(s"Unsupported scene type: $other"))

  protected def rebuildGeometry(
    specs: List[ObjectSpec],
    renderer: io.github.lene.optix.OptiXRenderer
  ): Unit =
    renderer.clearAllInstances()
    RenderModeSelector.classify(specs) match
      case _ if SceneGroups.hasMixedEdge4D(specs) =>
        buildMixedSceneObjects(specs, renderer)

      case SceneType.SimpleMixed(allSpecs, _) =>
        logger.debug(s"Rebuilding mixed scene: ${allSpecs.size} objects")
        buildMixedSceneObjects(allSpecs, renderer)

      case SceneType.Unsupported(_) =>
        sys.error("Complex mixed scenes not supported for rebuilding")

      case sceneType =>
        GeometryRegistry.builderFor(specs, textureDir) match
          case Some(builder) =>
            val effectiveMaxInstances = computeEffectiveMaxInstances(builder, specs)
            builder.validateAndBuild(specs, renderer, effectiveMaxInstances).get
          case None =>
            logger.warn(s"Cannot rebuild scene type: $sceneType")
            sys.error(s"Scene type $sceneType not supported for rebuilding")

  /** Builds each `SceneGroups.buildOrder` group with its own builder: cube-sponge specs need
    * CubeSpongeSceneBuilder (instance-explosion path), other triangle meshes
    * TriangleMeshSceneBuilder (H-sponge-showcase-crash fix), edge-rendered 4D objects
    * TesseractEdgeSceneBuilder -- first, since it may reinitialize the renderer. */
  private def buildMixedSceneObjects(
    specs: List[ObjectSpec],
    renderer: io.github.lene.optix.OptiXRenderer
  ): Unit =
    val groups = SceneGroups.buildOrder(specs)
    val effectiveMaxInstances = maxInstancesForGroups(groups)
    groups.foreach { group =>
      GeometryRegistry.builderFor(group, textureDir)
        .map(_.validateAndBuild(group, renderer, effectiveMaxInstances).get)
        .getOrElse {
          val types = group.map(_.objectType).distinct.mkString(", ")
          sys.error(s"No builder found for types: $types")
        }
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
