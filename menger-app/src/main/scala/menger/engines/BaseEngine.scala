package menger.engines

import java.util.concurrent.atomic.AtomicLong

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.Game
import com.typesafe.scalalogging.LazyLogging
import io.github.lene.optix.OptiXRenderer
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.dsl.Ambient
import menger.dsl.DenoiseMode
import menger.dsl.ResourceLimits
import menger.engines.scene.SceneBuilder
import menger.engines.scene.SceneCost
import menger.engines.scene.VramBudget
import menger.failure.Fail
import menger.input.OptiXRenderResources
import menger.optix.CameraState
import menger.optix.OptiXRendererWrapper
import menger.optix.SceneConfigurator
import menger.spec.ObjectSpec
import menger.spec.ObjectType

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
  // The ambient model to set; None leaves the renderer's own default (occlusion).
  protected def ambient: Option[Ambient] = None

  protected def configureOutputMode(renderer: OptiXRenderer): Unit =
    renderer.setDenoisingEnabled(denoiseMode == DenoiseMode.Final)
    if accumulationFrames > 1 then renderer.setAccumulationFrames(accumulationFrames)
    applyAmbient(renderer, ambient)

  protected def ambientFor(configs: SceneConverter.SceneConfigs): Option[Ambient] = configs.ambient

  // A frame's denoising: an engine with a command-line override applies it to every frame.
  protected def denoiseModeFor(configs: SceneConverter.SceneConfigs): DenoiseMode =
    configs.denoiseMode

  protected def applyAmbient(renderer: OptiXRenderer, model: Option[Ambient]): Unit =
    model.foreach(a => renderer.setAmbient(AmbientMapping.toModel(a)))

  // Override in concrete engines that need auto-adjustment (e.g. InteractiveEngine)
  protected def computeEffectiveMaxInstances(builder: SceneBuilder, specs: List[ObjectSpec]): Int =
    maxInstances

  /** The instance budget for `specs`: the configured `maxInstances`, or -- unless the user set
    * it explicitly -- twice what the scene needs when that exceeds it. */
  protected def autoAdjustedMaxInstances(
    builder: SceneBuilder,
    specs: List[ObjectSpec],
    userSetMaxInstances: Boolean
  ): Int =
    if userSetMaxInstances then maxInstances
    else
      val required = builder.calculateRequiredInstances(specs)
      val adjusted = BaseEngine.adjustedMaxInstances(required, maxInstances)
      if adjusted != maxInstances then
        logger.info(
          s"Auto-adjusting max instances: $maxInstances → $adjusted " +
          s"(scene requires $required)"
        )
      adjusted

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

  // Triangles of the scene the last passing check let through: still on the GPU when the next
  // check runs, and cleared before that scene is built (menger#179).
  private val residentTriangles = AtomicLong(0L)

  /** The device's free memory; None until the renderer exists, or when the query fails.
    * Overridable so a test can play a GPU of any size. */
  protected def freeGpuMemoryBytes: Option[Long] = rendererWrapper.freeGpuMemoryBytes

  protected def firstRenderPending: Boolean = !rendererWrapper.hasRendered

  private def gpuMemory: Option[VramBudget.GpuMemory] =
    freeGpuMemoryBytes.map(VramBudget.GpuMemory(_, residentTriangles.get, firstRenderPending))

  /** Refuses a scene over the scene-wide budget before any mesh is allocated, on every build
    * path -- start, reload, preview, animation (review 2026-10-14, menger#80/#107). The triangle
    * budget is what the free GPU memory takes (menger#179), the fixed
    * `ResourceLimits.sceneMaxTriangles` where it cannot be read; the other budgets are
    * `ResourceLimits`'. Caustics are left out: the validator checks them from the DSL scene. */
  protected def withinBudget(specs: List[ObjectSpec]): Try[Unit] =
    val memory = gpuMemory
    val budget = memory.fold(ResourceLimits.sceneMaxTriangles)(VramBudget.triangleBudget)
    val estimate =
      SceneCost.estimate(specs, accumulationFrames, SceneCost.CausticsLoad.None, budget)
    val overTriangleBudget = estimate.triangles > budget
    SceneCost.check(estimate)
      .left.map(problem =>
        IllegalArgumentException(problem + budgetOrigin(memory.filter(_ => overTriangleBudget))))
      .map(_ => residentTriangles.set(estimate.triangles))
      .toTry

  // Where a refusal's triangle budget came from, so "budget 14,000,000" can be traced to the GPU
  private def budgetOrigin(memory: Option[VramBudget.GpuMemory]): String =
    memory.fold("")(m =>
      s" [the triangle budget follows the free GPU memory: ${m.freeBytes / VramBudget.BytesPerMiB} MiB free]")

  protected def buildSceneFromSpecs(
    specs: List[ObjectSpec],
    renderer: io.github.lene.optix.OptiXRenderer
  ): Try[Unit] =
    withinBudget(specs).flatMap(_ => buildSpecsWithinBudget(specs, renderer))

  private def buildSpecsWithinBudget(
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
    withinBudget(specs).flatMap(_ => buildConfigSpecsWithinBudget(specs, renderer))

  private def buildConfigSpecsWithinBudget(
    specs: List[ObjectSpec],
    renderer: io.github.lene.optix.OptiXRenderer
  ): Try[Unit] =
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
    withinBudget(specs).get // before clearing, so a refused scene leaves the old one standing
    renderer.clearAllInstances()
    RenderModeSelector.classify(specs) match
      case _ if SceneGroups.hasMixedEdge4D(specs) =>
        buildMixedSceneObjects(specs, renderer)

      case SceneType.SimpleMixed(allSpecs, _) =>
        logger.debug(s"Rebuilding mixed scene: ${allSpecs.size} objects")
        buildMixedSceneObjects(allSpecs, renderer)

      case SceneType.Unsupported(_) =>
        Fail.unsupported("rebuilding a complex mixed scene")

      case sceneType =>
        GeometryRegistry.builderFor(specs, textureDir) match
          case Some(builder) =>
            val effectiveMaxInstances = computeEffectiveMaxInstances(builder, specs)
            builder.validateAndBuild(specs, renderer, effectiveMaxInstances).get
          case None =>
            logger.warn(s"Cannot rebuild scene type: $sceneType")
            Fail.unsupported(s"rebuilding a scene of type $sceneType")

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
          Fail.unsupported(s"a scene builder for the types $types")
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

object BaseEngine:

  // The configured budget, or twice the scene's requirement (capped) when that exceeds it.
  def adjustedMaxInstances(required: Int, configured: Int): Int =
    if required > 0 && required > configured then
      Math.min(required * 2, menger.common.Const.maxInstancesLimit)
    else configured
