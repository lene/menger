package menger.engines

import java.util.concurrent.atomic.AtomicReference

import scala.collection.mutable.ArrayBuffer
import scala.util.Try

import com.badlogic.gdx.graphics.GL20
import com.typesafe.scalalogging.LazyLogging
import menger.ObjectSpec
import menger.ProfilingConfig
import menger.Projection4DSpec
import menger.RotationProjectionParameters
import menger.common.Const
import menger.common.ImageSize
import menger.common.ObjectType
import menger.config.OptiXEngineConfig
import menger.engines.scene.SceneBuilder
import menger.engines.scene.TriangleMeshSceneBuilder
import menger.input.EventDispatcher
import menger.input.GdxRuntime
import menger.input.Observer
import menger.input.OptiXCameraHandler
import menger.input.OptiXInputMultiplexer
import menger.input.OptiXKeyHandler
import menger.objects.higher_d.Projection
import menger.objects.higher_d.TesseractSponge2Mesh
import menger.objects.higher_d.TesseractSpongeMesh
import menger.optix.CameraState
import menger.optix.SceneConfigurator

class InteractiveEngine(
  config: OptiXEngineConfig,
  userSetMaxInstances: Boolean = false
)(using ProfilingConfig)
    extends BaseEngine(config.execution.maxInstances)
    with TimeoutSupport with LazyLogging with SavesScreenshots with Observer:

  // Convenience accessors for config sections
  private val scene       = config.scene
  private val camera      = config.camera
  private val environment = config.environment
  private val execution   = config.execution

  override protected def textureDir: String = execution.textureDir

  override protected def renderConfig: menger.optix.RenderConfig = config.render

  // Required by TimeoutSupport trait
  override def timeout: Float = execution.timeout

  // Extract object specs from config (must be provided)
  private val objectSpecs: List[ObjectSpec] = scene.objectSpecs.getOrElse {
    logger.error(
      "SceneConfig must provide objectSpecs. " +
      "Legacy single-object parameters are no longer supported."
    )
    sys.error("SceneConfig must provide objectSpecs")
  }

  // Event dispatcher for 4D rotation events
  private val eventDispatcher = EventDispatcher().withObserver(this)

  private val currentObjectSpecs =
    new AtomicReference[Option[List[ObjectSpec]]](Some(objectSpecs))

  // Keyboard handler for 4D rotation (initialized in finalizeCreate)
  private val keyHandler = new AtomicReference[Option[OptiXKeyHandler]](None)

  // Track if we have 4D projected objects (need rebuild on rotation)
  private lazy val has4DObjects: Boolean =
    currentObjectSpecs.get().exists(_.exists(spec => ObjectType.isProjected4D(spec.objectType)))

  /** Cached per-spec slot mapping enabling the GPU 4D-rotation fast path.
    * Populated by `buildScene4DTracked` when the scene is 4D-only and
    * `gpuProject4D` is on; cleared on any rebuild path that does not record
    * slot indices. See `WithAnimation.Anim4DState` for the symmetric animation
    * fast path. */
  private val anim4DState: AtomicReference[Option[WithAnimation.Anim4DState]] =
    new AtomicReference(None)

  override protected val sceneConfigurator: SceneConfigurator = SceneConfigurator(
    camera.position,
    camera.lookAt,
    camera.up,
    environment.lights
  )

  override protected val cameraState: CameraState =
    CameraState(camera.position, camera.lookAt, camera.up)

  private lazy val cameraController: OptiXCameraHandler =
    OptiXCameraHandler(rendererWrapper, cameraState, renderResources,
      camera.position, camera.lookAt, camera.up, eventDispatcher)

  // Handle rotation/projection events from keyboard and mouse
  override def handleEvent(event: RotationProjectionParameters): Unit =
    logger.debug(
      s"Received rotation event: rotXW=${event.rotXW}, rotYW=${event.rotYW}, rotZW=${event.rotZW}"
    )
    if has4DObjects then
      val updatedSpecs = currentObjectSpecs.get().map(_.map { spec =>
        if ObjectType.isProjected4D(spec.objectType) then
          val currentProj = spec.projection4D.getOrElse(Projection4DSpec.default)
          val newEyeW =
            // defaultEyeW acts as a sentinel: a rotation-only event leaves eyeW unchanged.
            // A user who genuinely wants to reset eyeW to exactly defaultEyeW cannot do so here.
            if event.eyeW != Const.defaultEyeW then
              val updatedProjection =
                Projection(currentProj.eyeW, currentProj.screenW) + event.projection
              updatedProjection.eyeW
            else
              currentProj.eyeW
          val newProj = currentProj.copy(
            eyeW = newEyeW,
            rotXW = currentProj.rotXW + event.rotXW,
            rotYW = currentProj.rotYW + event.rotYW,
            rotZW = currentProj.rotZW + event.rotZW
          )
          logger.debug(
            s"Updated tesseract: rotXW=${newProj.rotXW}, rotYW=${newProj.rotYW}, " +
            s"rotZW=${newProj.rotZW}, eyeW=${newProj.eyeW}"
          )
          spec.copy(projection4D = Some(newProj))
        else
          spec
      })
      currentObjectSpecs.set(updatedSpecs)

      val fastPathTaken = updatedSpecs.exists(specs =>
        tryRotation4DFastPath(specs, rendererWrapper.renderer)
      )
      if !fastPathTaken then
        rebuildScene()
      renderResources.markNeedsRender()
      GdxRuntime.requestRendering()

  /** Per-mesh GPU 4D-projection update path. Mirrors `WithAnimation.tryAnim4DFastPath`.
    * Returns true iff every spec is 4D-projected, slot mapping is cached, and the
    * change is purely a `Projection4DSpec` delta — in which case we call
    * `renderer.updateMesh4DProjection` per slot and skip the full geometry rebuild
    * (whose multi-object path hangs, per H-mixed-frac-int-interactive-hang). */
  private def tryRotation4DFastPath(
    newSpecs: List[ObjectSpec],
    renderer: menger.optix.OptiXRenderer
  ): Boolean =
    if !renderConfig.gpuProject4D then false
    else anim4DState.get match
      case None => false
      case Some(prev) =>
        if !WithAnimation.specsDifferOnlyIn4DProjection(prev.specs, newSpecs) then false
        else
          prev.specs.lazyZip(newSpecs).lazyZip(prev.slotsPerSpec).foreach {
            case (prevSpec, newSpec, slots) =>
              if prevSpec.projection4D != newSpec.projection4D then
                val proj = newSpec.projection4D.getOrElse(Projection4DSpec.default)
                slots.foreach { slot =>
                  renderer.updateMesh4DProjection(
                    slot,
                    eyeW = proj.eyeW, screenW = proj.screenW,
                    rotXW = proj.rotXW, rotYW = proj.rotYW, rotZW = proj.rotZW
                  )
                }
          }
          anim4DState.set(Some(prev.copy(specs = newSpecs)))
          true

  // Level thresholds for warnings (based on triangle counts and performance)
  private val VolumeLevelWarning = Const.Engine.spongeLevelWarningThreshold
  private val SurfaceLevelWarning = Const.Engine.spongeLevelWarningThreshold
  private val VolumeLevelMax = Const.Engine.cubeSpongeMaxLevel
  private val SurfaceLevelMax = Const.Engine.cubeSpongeMaxLevel

  private def warnIfHighLevel(spec: ObjectSpec): Unit =
    spec.level.foreach { level =>
      val intLevel = level.toInt
      spec.objectType match
        case "sponge-volume" =>
          if intLevel >= VolumeLevelWarning then
            val estimatedTriangles =
              math.pow(Const.Engine.cubesPerSpongeLevel, intLevel).toLong *
              Const.Engine.trianglesPerCube
            logger.warn(
              s"Sponge level $intLevel may be slow " +
              s"(~${estimatedTriangles / 1000}K triangles)"
            )
          if intLevel > VolumeLevelMax then
            logger.error(s"Sponge level $intLevel exceeds recommended maximum ($VolumeLevelMax)")
        case "sponge-surface" =>
          if intLevel >= SurfaceLevelWarning then
            val estimatedTriangles =
              math.pow(Const.Engine.trianglesPerCube, intLevel).toLong * 6 * 2
            logger.warn(
              s"Sponge level $intLevel may be slow " +
              s"(~${estimatedTriangles / 1000}K triangles)"
            )
          if intLevel > SurfaceLevelMax then
            logger.error(s"Sponge level $intLevel exceeds recommended maximum ($SurfaceLevelMax)")
        case "tesseract-sponge" =>
          if intLevel >= Const.Engine.tesseractSpongeWarnLevel then
            val triangles = TesseractSpongeMesh.estimatedTriangles(intLevel)
            logger.warn(
              s"tesseract-sponge level $intLevel may be slow " +
              s"(~${triangles / 1000}K triangles)"
            )
          if intLevel > Const.Engine.tesseractSpongeMaxLevel then
            logger.error(
              s"tesseract-sponge level $intLevel exceeds recommended maximum " +
              s"(${Const.Engine.tesseractSpongeMaxLevel})"
            )
        case "tesseract-sponge-2" =>
          if intLevel >= Const.Engine.tesseractSponge2WarnLevel then
            val triangles = TesseractSponge2Mesh.estimatedTriangles(intLevel)
            logger.warn(
              s"tesseract-sponge-2 level $intLevel may be slow " +
              s"(~${triangles / 1000}K triangles)"
            )
          if intLevel > Const.Engine.tesseractSponge2MaxLevel then
            logger.error(
              s"tesseract-sponge-2 level $intLevel exceeds recommended maximum " +
              s"(${Const.Engine.tesseractSponge2MaxLevel})"
            )
        case _ => // No warning for other types
    }

  // Override BaseEngine default: auto-adjust maxInstances when user did not set it explicitly
  override protected def computeEffectiveMaxInstances(
    builder: SceneBuilder,
    specs: List[ObjectSpec]
  ): Int =
    if userSetMaxInstances then
      execution.maxInstances
    else
      val required = builder.calculateRequiredInstances(specs)
      if required > 0 && required > execution.maxInstances then
        val adjusted = Math.min(required * 2, menger.common.Const.maxInstancesLimit)
        logger.info(
          s"Auto-adjusting max instances: ${execution.maxInstances} → $adjusted " +
          s"(scene requires $required)"
        )
        adjusted
      else
        execution.maxInstances

  override def create(): Unit =
    logger.info(s"Creating InteractiveEngine with ${objectSpecs.length} objects")
    objectSpecs.foreach(warnIfHighLevel)
    val renderer = rendererWrapper.renderer
    // Resize the native instance buffer up-front when auto-adjust says we need
    // more than the constructor-time budget — otherwise large cube-sponge / mixed
    // scenes silently drop instances past the cap.
    val requiredMax = requiredMaxInstancesFor(objectSpecs)
    if requiredMax > execution.maxInstances then
      renderer.reinitialize(requiredMax)
    sceneConfigurator.configureLights(renderer)
    sceneConfigurator.configurePlanes(renderer, environment.planes)
    sceneConfigurator.configureCamera(renderer)
    buildScene4DTrackedOrFallback(objectSpecs, renderer)
      .flatMap { _ =>
        Try {
          renderer.setRenderConfig(renderConfig)
          renderer.setCausticsConfig(config.caustics)
          environment.background.foreach(sceneConfigurator.setBackgroundColor(renderer, _))
          finalizeCreate()
        }
      }
      .recover { case e: Exception =>
        logger.error(s"Failed to create OptiX scene: ${e.getMessage}", e)
        GdxRuntime.exit()
      }.get

  private def resetTo4DDefaults(): Unit =
    logger.debug("Resetting 4D view to initial state")
    currentObjectSpecs.set(Some(objectSpecs))
    rebuildScene()
    renderResources.markNeedsRender()
    GdxRuntime.requestRendering()

  private def finalizeCreate(): Unit =
    val handler = OptiXKeyHandler(eventDispatcher, onReset = () => resetTo4DDefaults())
    keyHandler.set(Some(handler))
    GdxRuntime.setInputProcessor(OptiXInputMultiplexer(cameraController, handler))
    GdxRuntime.setContinuousRendering(false)
    GdxRuntime.requestRendering()
    if execution.timeout > 0 then startExitTimer(execution.timeout)

  /** Build the initial scene; if it is 4D-only triangle meshes and `gpuProject4D`
    * is on, capture per-spec slot indices so subsequent rotation events can hit
    * the fast path. Otherwise fall back to the generic build and clear cached
    * fast-path state. Mirrors `WithAnimation.buildAnim4DTrackedOrFallback`. */
  private def buildScene4DTrackedOrFallback(
    specs: List[ObjectSpec],
    renderer: menger.optix.OptiXRenderer
  ): Try[Unit] =
    if renderConfig.gpuProject4D && WithAnimation.is4DOnlyTriangleMeshScene(specs) then
      val slotsBuf: scala.collection.mutable.Map[Int, ArrayBuffer[Int]] =
        scala.collection.mutable.Map.empty
      val builder = TriangleMeshSceneBuilder(
        textureDir, gpuProject4D = true,
        mesh4DRecorder = (specIdx: Int, slotIdx: Int) =>
          slotsBuf.getOrElseUpdate(specIdx, ArrayBuffer.empty[Int]) += slotIdx
      )(using profilingConfig)
      val effectiveMaxInstances = computeEffectiveMaxInstances(builder, specs)
      val result = builder.buildScene(specs, renderer, effectiveMaxInstances)
      result.foreach { _ =>
        if slotsBuf.size == specs.size then
          val slotsPerSpec = specs.indices.map(i =>
            slotsBuf.getOrElse(i, ArrayBuffer.empty[Int]).toVector
          ).toVector
          anim4DState.set(Some(WithAnimation.Anim4DState(specs, slotsPerSpec)))
        else
          anim4DState.set(None)
      }
      result.recover { case _ => anim4DState.set(None) }
      result
    else
      anim4DState.set(None)
      buildSceneFromSpecs(specs, renderer)

  private def rebuildScene(): Unit =
    currentObjectSpecs.get() match
      case Some(specs) =>
        val renderer    = rendererWrapper.renderer
        val savedEye    = cameraController.currentEye
        val savedLookAt = cameraController.currentLookAt
        val savedUp     = cameraController.currentUp
        logger.debug(
          s"Rebuilding scene with updated rotation; camera: eye=$savedEye, lookAt=$savedLookAt"
        )
        Try {
          rebuildGeometry(specs, renderer)
          cameraState.updateCamera(renderer, savedEye, savedLookAt, savedUp)
          logger.debug("Scene rebuild complete")
        }.recover { case e =>
          logger.error(s"Failed to rebuild scene: ${e.getMessage}", e)
        }
      case None =>
        logger.warn("Cannot rebuild single-object scene interactively")

  override def render(): Unit =
    GdxRuntime.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)
    keyHandler.get().foreach(_.update(GdxRuntime.deltaTime))
    val width  = GdxRuntime.width
    val height = GdxRuntime.height
    if width > 0 && height > 0 then
      if renderResources.currentDimensions.isEmpty then
        cameraState.updateCameraAspectRatio(rendererWrapper.renderer, ImageSize(width, height))
        renderResources.markNeedsRender()
      if renderResources.needsRender then
        val size = ImageSize(width, height)
        val rgbaBytes =
          if execution.enableStats then renderWithStats(width, height)
          else rendererWrapper.renderScene(size)
        renderResources.renderToScreen(rgbaBytes, width, height)
      else
        renderResources.redrawExisting(width, height)
      saveImage()
    if shouldExitAfterSave then
      renderResources.markSaved()
      GdxRuntime.exit()

  private def shouldExitAfterSave: Boolean =
    execution.saveName.isDefined && !renderResources.hasSaved && execution.timeout == 0

  override protected def currentSaveName: Option[String] = execution.saveName
  override protected def allowUniformRender: Boolean = execution.allowUniformRender

  private def renderWithStats(width: Int, height: Int): Array[Byte] =
    val result = rendererWrapper.renderSceneWithStats(ImageSize(width, height))
    val stats  = result.stats
    logger.info(
      s"Ray stats: primary=${stats.primaryRays} total=${stats.totalRays} " +
      s"reflected=${stats.reflectedRays} refracted=${stats.refractedRays} " +
      s"shadow=${stats.shadowRays} aa=${stats.aaRays} " +
      s"depth=${stats.minDepthReached}-${stats.maxDepthReached}"
    )
    result.image

  override def resize(width: Int, height: Int): Unit = {}

  override def dispose(): Unit =
    logger.debug("Disposing InteractiveEngine")
    super.dispose()
