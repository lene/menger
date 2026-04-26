package menger.engines

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.Game
import com.typesafe.scalalogging.LazyLogging
import menger.ObjectSpec
import menger.OptiXRenderResources
import menger.ProfilingConfig
import menger.common.ObjectType
import menger.common.ValidationException
import menger.dsl.SceneConverter
import menger.engines.scene.SceneBuilder
import menger.engines.scene.SphereSceneBuilder
import menger.engines.scene.TesseractEdgeSceneBuilder
import menger.engines.scene.TriangleMeshSceneBuilder
import menger.optix.CameraState
import menger.optix.OptiXRendererWrapper
import menger.optix.RenderConfig
import menger.optix.SceneConfigurator

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

  protected def selectMeshBuilder(specs: List[ObjectSpec]): SceneBuilder =
    val all4DProjected = specs.forall(s => ObjectType.isProjected4D(s.objectType))
    val hasEdgeRendering = specs.exists(_.hasEdgeRendering)
    if all4DProjected && hasEdgeRendering then
      TesseractEdgeSceneBuilder(textureDir)(using profilingConfig)
    else
      TriangleMeshSceneBuilder(textureDir, renderConfig.gpuProject4D)(using profilingConfig)

  // Must be provided by concrete engine — where texture assets live
  protected def textureDir: String

  protected def buildSceneFromSpecs(
    specs: List[ObjectSpec],
    renderer: menger.optix.OptiXRenderer
  ): Try[Unit] =
    SceneClassifier.classify(specs) match
      case SceneType.SimpleMixed(allSpecs, _) =>
        val sphereSpecs = allSpecs.filter(_.objectType.toLowerCase == "sphere")
        val meshSpecs   = allSpecs.filterNot(_.objectType.toLowerCase == "sphere")
        logger.info(s"Mixed scene: ${sphereSpecs.size} spheres + ${meshSpecs.size} mesh objects")
        Try(buildMixedSceneObjects(sphereSpecs, meshSpecs, renderer))

      case SceneType.ComplexMixed(allSpecs) =>
        val objectTypes = allSpecs.map(_.objectType).distinct
        Failure(UnsupportedOperationException(
          "Cannot mix spheres with multiple different triangle mesh types. " +
          s"Objects: ${objectTypes.mkString(", ")}. " +
          "Spheres can be mixed with one mesh type at a time."
        ))

      case sceneType =>
        SceneClassifier.selectSceneBuilder(sceneType, Some(textureDir), renderConfig.gpuProject4D) match
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
    renderer: menger.optix.OptiXRenderer
  ): Try[Unit] =
    val specs = configs.scene.objectSpecs.getOrElse(List.empty)
    val sceneType = SceneClassifier.classify(specs)
    sceneType match
      case SceneType.Spheres(_) =>
        SphereSceneBuilder().buildScene(specs, renderer, maxInstances)
      case SceneType.TriangleMeshes(_) =>
        SceneClassifier.selectSceneBuilder(sceneType, Some(textureDir), renderConfig.gpuProject4D) match
          case Some(builder) => builder.buildScene(specs, renderer, maxInstances)
          case None          => Failure(UnsupportedOperationException(s"No builder for $sceneType"))
      case SceneType.SimpleMixed(allSpecs, _) =>
        Try {
          val sphereSpecs = allSpecs.filter(_.objectType.toLowerCase == "sphere")
          val meshSpecs   = allSpecs.filterNot(_.objectType.toLowerCase == "sphere")
          if sphereSpecs.nonEmpty then
            SphereSceneBuilder().buildScene(sphereSpecs, renderer, maxInstances).get
          if meshSpecs.nonEmpty then
            SceneClassifier.selectSceneBuilder(
              SceneType.TriangleMeshes(meshSpecs), Some(textureDir)
            ) match
              case Some(builder) => builder.buildScene(meshSpecs, renderer, maxInstances).get
              case None =>
                val types = meshSpecs.map(_.objectType).distinct.mkString(", ")
                sys.error(s"No mesh builder found for types: $types")
        }
      case other =>
        SceneClassifier.selectSceneBuilder(other, Some(textureDir)) match
          case Some(builder) => builder.buildScene(specs, renderer, maxInstances)
          case None          =>
            Failure(UnsupportedOperationException(s"Unsupported scene type: $other"))

  protected def rebuildGeometry(
    specs: List[ObjectSpec],
    renderer: menger.optix.OptiXRenderer
  ): Unit =
    renderer.clearAllInstances()
    SceneClassifier.classify(specs) match
      case SceneType.SimpleMixed(allSpecs, _) =>
        val sphereSpecs = allSpecs.filter(_.objectType.toLowerCase == "sphere")
        val meshSpecs   = allSpecs.filterNot(_.objectType.toLowerCase == "sphere")
        logger.debug(
          s"Rebuilding mixed scene: ${sphereSpecs.size} spheres + ${meshSpecs.size} mesh objects"
        )
        buildMixedSceneObjects(sphereSpecs, meshSpecs, renderer)

      case SceneType.ComplexMixed(_) =>
        sys.error("Complex mixed scenes not supported for rebuilding")

      case sceneType =>
        SceneClassifier.selectSceneBuilder(sceneType, Some(textureDir), renderConfig.gpuProject4D) match
          case Some(builder) =>
            builder.buildScene(specs, renderer, maxInstances).get
          case None =>
            logger.warn(s"Cannot rebuild scene type: $sceneType")
            sys.error(s"Scene type $sceneType not supported for rebuilding")

  private def buildMixedSceneObjects(
    sphereSpecs: List[ObjectSpec],
    meshSpecs: List[ObjectSpec],
    renderer: menger.optix.OptiXRenderer
  ): Unit =
    val meshBuilder = selectMeshBuilder(meshSpecs)
    val effectiveMaxInstances = computeEffectiveMaxInstances(meshBuilder, meshSpecs)
    if sphereSpecs.nonEmpty then
      SphereSceneBuilder().buildScene(sphereSpecs, renderer, effectiveMaxInstances).get
    if meshSpecs.nonEmpty then
      meshBuilder.buildScene(meshSpecs, renderer, effectiveMaxInstances).get

  // Default lifecycle — concrete engines override what they need
  override def create(): Unit = {}
  override def render(): Unit = {}
  override def resize(width: Int, height: Int): Unit = {}
  override def dispose(): Unit =
    renderResources.dispose()
    rendererWrapper.dispose()
  override def pause(): Unit  = {}
  override def resume(): Unit = {}
