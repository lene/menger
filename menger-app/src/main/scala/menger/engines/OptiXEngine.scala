package menger.engines

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.GL20
import com.typesafe.scalalogging.LazyLogging
import menger.ObjectSpec
import menger.OptiXRenderResources
import menger.ProfilingConfig
import menger.Projection4DSpec
import menger.RotationProjectionParameters
import menger.common.Const
import menger.common.ImageSize
import menger.common.ObjectType
import menger.common.ValidationException
import menger.config.OptiXEngineConfig
import menger.engines.scene.CubeSpongeSceneBuilder
import menger.engines.scene.SceneBuilder
import menger.engines.scene.SphereSceneBuilder
import menger.engines.scene.TriangleMeshSceneBuilder
import menger.input.EventDispatcher
import menger.input.Observer
import menger.input.OptiXCameraHandler
import menger.input.OptiXInputMultiplexer
import menger.input.OptiXKeyHandler
import menger.optix.CameraState
import menger.optix.OptiXRendererWrapper
import menger.optix.SceneConfigurator

enum SceneType:
  case CubeSponges(specs: List[ObjectSpec])
  case Spheres(specs: List[ObjectSpec])
  case TriangleMeshes(specs: List[ObjectSpec])
  case Mixed(specs: List[ObjectSpec])

class OptiXEngine(config: OptiXEngineConfig)(using profilingConfig: ProfilingConfig)
  extends RenderEngine with TimeoutSupport with LazyLogging with SavesScreenshots with Observer:

  // Convenience accessors for config sections
  private val scene = config.scene
  private val camera = config.camera
  private val environment = config.environment
  private val execution = config.execution

  // Required by TimeoutSupport trait
  override def timeout: Float = execution.timeout

  // Extract object specs from config (must be provided)
  private val objectSpecs: List[ObjectSpec] = scene.objectSpecs.getOrElse {
    logger.error("SceneConfig must provide objectSpecs. Legacy single-object parameters are no longer supported.")
    // Wartremover: Use sys.error instead of throw for disabled features
    sys.error("SceneConfig must provide objectSpecs")
  }

  // Event dispatcher for 4D rotation events
  private val eventDispatcher = EventDispatcher().withObserver(this)

  // Mutable state for current object specs (for interactive rotation updates)
  // Wartremover: var required for interactive rotation state management
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var currentObjectSpecs: Option[List[ObjectSpec]] = Some(objectSpecs)

  // Track if we have tesseract objects (need rebuild on rotation)
  private lazy val hasTesseracts: Boolean = currentObjectSpecs.exists(_.exists(_.objectType == "tesseract"))

  // Handle rotation events from keyboard
  override def handleEvent(event: RotationProjectionParameters): Unit =
    logger.debug(s"Received rotation event: rotXW=${event.rotXW}, rotYW=${event.rotYW}, rotZW=${event.rotZW}")
    if hasTesseracts then
      // Update object specs with new rotation values
      currentObjectSpecs = currentObjectSpecs.map(_.map { spec =>
        if spec.objectType == "tesseract" then
          val currentProj = spec.projection4D.getOrElse(Projection4DSpec.default)
          val newProj = currentProj.copy(
            rotXW = currentProj.rotXW + event.rotXW,
            rotYW = currentProj.rotYW + event.rotYW,
            rotZW = currentProj.rotZW + event.rotZW
          )
          logger.debug(s"Updated tesseract rotation: rotXW=${newProj.rotXW}, rotYW=${newProj.rotYW}, rotZW=${newProj.rotZW}")
          spec.copy(projection4D = Some(newProj))
        else
          spec
      })
      // Rebuild scene with updated rotation
      rebuildScene()
      // Mark resources as needing render and request it
      renderResources.markNeedsRender()
      Gdx.graphics.requestRendering()

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
            // 12^level sub-faces * 6 faces * 2 triangles
            val estimatedTriangles =
              math.pow(Const.Engine.trianglesPerCube, intLevel).toLong * 6 * 2
            logger.warn(
              s"Sponge level $intLevel may be slow " +
              s"(~${estimatedTriangles / 1000}K triangles)"
            )
          if intLevel > SurfaceLevelMax then
            logger.error(s"Sponge level $intLevel exceeds recommended maximum ($SurfaceLevelMax)")
        case _ => // No warning for other types
    }

  // Composition: Three focused components instead of one god object
  private val rendererWrapper = OptiXRendererWrapper(execution.maxInstances)
  private val sceneConfigurator = SceneConfigurator(
    Failure(UnsupportedOperationException("Legacy geometry generator removed. Use objectSpecs instead.")),
    camera.position,
    camera.lookAt,
    camera.up,
    environment.plane,
    environment.lights
  )
  private val cameraState = CameraState(camera.position, camera.lookAt, camera.up)

  private val renderResources: OptiXRenderResources = OptiXRenderResources(0, 0)
  private lazy val cameraController: OptiXCameraHandler =
    OptiXCameraHandler(rendererWrapper, cameraState, renderResources,
      camera.position, camera.lookAt, camera.up, eventDispatcher)

  override def create(): Unit =
    val result = createMultiObjectScene(objectSpecs)

    result.recover { case e: Exception =>
      logger.error(s"Failed to create OptiX scene: ${e.getMessage}", e)
      Gdx.app.exit()
    }.get  // Intentional .get - initialization failure should crash (documented)

  private def classifyScene(specs: List[ObjectSpec]): SceneType =
    val objectTypes = specs.map(_.objectType).distinct

    if objectTypes.contains("cube-sponge") then
      SceneType.CubeSponges(specs)
    else if objectTypes.forall(_ == "sphere") then
      SceneType.Spheres(specs)
    else if objectTypes.forall(isTriangleMeshType) then
      SceneType.TriangleMeshes(specs)
    else
      SceneType.Mixed(specs)

  private def isTriangleMeshType(objectType: String): Boolean =
    objectType == "cube" || ObjectType.isSponge(objectType) || ObjectType.isHypercube(objectType)

  private def selectSceneBuilder(sceneType: SceneType): Option[SceneBuilder] =
    sceneType match
      case SceneType.Spheres(_) => Some(SphereSceneBuilder())
      case SceneType.TriangleMeshes(_) => Some(TriangleMeshSceneBuilder(execution.textureDir)(using profilingConfig))
      case SceneType.CubeSponges(_) => Some(CubeSpongeSceneBuilder())
      case SceneType.Mixed(_) => None

  private def createMultiObjectScene(specs: List[ObjectSpec]): Try[Unit] =
    logger.info(s"Creating OptiXEngine with ${specs.length} objects")

    // Warn about high sponge levels before generating geometry
    specs.foreach(warnIfHighLevel)

    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureLights(renderer)
    sceneConfigurator.configurePlane(renderer)
    sceneConfigurator.configureCamera(renderer)

    // Determine scene type and setup using strategy pattern
    val result = classifyScene(specs) match
      case sceneType @ SceneType.Mixed(specs) =>
        val objectTypes = specs.map(_.objectType).distinct
        Failure(UnsupportedOperationException(
          "Cannot mix spheres and triangle meshes in the same scene yet. " +
          s"Objects: ${objectTypes.mkString(", ")}"
        ))

      case sceneType =>
        selectSceneBuilder(sceneType) match
          case Some(builder) =>
            builder.validate(specs, execution.maxInstances) match
              case Left(error) => Failure(ValidationException(error, "objectSpecs", specs.map(_.objectType)))
              case Right(_) => builder.buildScene(specs, renderer)
          case None =>
            Failure(UnsupportedOperationException(s"No builder available for $sceneType"))

    result.flatMap { _ =>
      Try {
        renderer.setRenderConfig(config.render)
        renderer.setCausticsConfig(config.caustics)
        environment.planeColor.foreach(sceneConfigurator.setPlaneColor(renderer, _))
        finalizeCreate()
      }
    }

  private def finalizeCreate(): Unit =
    // Register input multiplexer for mouse-based camera control and keyboard shortcuts
    val keyHandler = OptiXKeyHandler(eventDispatcher)
    Gdx.input.setInputProcessor(OptiXInputMultiplexer(cameraController, keyHandler))

    // Disable continuous rendering - we'll request renders only when needed
    Gdx.graphics.setContinuousRendering(false)
    Gdx.graphics.requestRendering()

    if execution.timeout > 0 then startExitTimer(execution.timeout)

  private def rebuildScene(): Unit =
    currentObjectSpecs match
      case Some(specs) =>
        Try {
          logger.debug("Rebuilding scene with updated rotation")

          val renderer = rendererWrapper.renderer

          // Save current camera state before dispose (which wipes everything)
          val savedEye = cameraController.currentEye
          val savedLookAt = cameraController.currentLookAt
          val savedUp = cameraController.currentUp
          logger.debug(s"Saving camera state: eye=$savedEye, lookAt=$savedLookAt")

          // Dispose and re-initialize renderer
          renderer.dispose()
          renderer.initialize(execution.maxInstances)

          // Recreate scene configuration
          sceneConfigurator.configureLights(renderer)
          sceneConfigurator.configurePlane(renderer)
          renderer.setRenderConfig(config.render)
          renderer.setCausticsConfig(config.caustics)
          environment.planeColor.foreach(sceneConfigurator.setPlaneColor(renderer, _))

          // Restore camera to saved position
          cameraState.updateCamera(renderer, savedEye, savedLookAt, savedUp)
          logger.debug(s"Restored camera state: eye=$savedEye, lookAt=$savedLookAt")

          // Rebuild geometry based on scene type using strategy pattern
          classifyScene(specs) match
            case sceneType =>
              selectSceneBuilder(sceneType) match
                case Some(builder) =>
                  builder.buildScene(specs, renderer).get
                case None =>
                  logger.warn("Cannot rebuild mixed scene type")
                  Failure(UnsupportedOperationException("Mixed scenes not supported for rebuilding")).get

          logger.debug("Scene rebuild complete")
        }.recover { case e =>
          logger.error(s"Failed to rebuild scene: ${e.getMessage}", e)
        }
      case None =>
        logger.warn("Cannot rebuild single-object scene interactively")

  override def render(): Unit =
    Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)

    val width = Gdx.graphics.getWidth
    val height = Gdx.graphics.getHeight

    // Only proceed if window dimensions are valid
    if width > 0 && height > 0 then
      // Initialize camera on first render (window is not resizable in OptiX mode)
      if renderResources.currentDimensions.isEmpty then
        cameraState.updateCameraAspectRatio(rendererWrapper.renderer, ImageSize(width, height))
        renderResources.markNeedsRender()

      // Only render the scene if something changed (camera moved, etc.)
      if renderResources.needsRender then
        val size = ImageSize(width, height)
        val rgbaBytes = if execution.enableStats then renderWithStats(width, height) else rendererWrapper.renderScene(size)
        renderResources.renderToScreen(rgbaBytes, width, height)
      else
        // Just redraw the existing texture without re-rendering
        renderResources.redrawExisting(width, height)

      saveImage()

    // Exit after saving when in non-interactive mode (unless timeout is set)
    if shouldExitAfterSave then
      renderResources.markSaved()
      Gdx.app.exit()

  private def shouldExitAfterSave: Boolean =
    execution.saveName.isDefined && !renderResources.hasSaved && execution.timeout == 0

  protected def currentSaveName: Option[String] = execution.saveName

  private def renderWithStats(width: Int, height: Int): Array[Byte] =
    val result = rendererWrapper.renderSceneWithStats(ImageSize(width, height))
    val stats = result.stats
    logger.info(
      s"Ray stats: primary=${stats.primaryRays} total=${stats.totalRays} " +
      s"reflected=${stats.reflectedRays} refracted=${stats.refractedRays} " +
      s"shadow=${stats.shadowRays} aa=${stats.aaRays} " +
      s"depth=${stats.minDepthReached}-${stats.maxDepthReached}"
    )
    result.image

  // Window resize is disabled for OptiX mode (setResizable(false) in Main)
  // This method is kept for interface compatibility but should never be called with different dimensions
  override def resize(width: Int, height: Int): Unit = {}

  override def dispose(): Unit =
    logger.debug("Disposing OptiXEngine")
    renderResources.dispose()
    rendererWrapper.dispose()

  override def pause(): Unit = {}
  override def resume(): Unit = {}
