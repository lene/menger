package menger.engines

import java.util.concurrent.atomic.AtomicReference

import scala.util.Failure
import scala.util.Try

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
import menger.engines.scene.TesseractEdgeSceneBuilder
import menger.engines.scene.TriangleMeshSceneBuilder
import menger.gdx.GdxRuntime
import menger.input.EventDispatcher
import menger.input.Observer
import menger.input.OptiXCameraHandler
import menger.input.OptiXInputMultiplexer
import menger.input.OptiXKeyHandler
import menger.objects.higher_d.Projection
import menger.objects.higher_d.TesseractSponge2Mesh
import menger.objects.higher_d.TesseractSpongeMesh
import menger.optix.CameraState
import menger.optix.OptiXRendererWrapper
import menger.optix.SceneConfigurator

class OptiXEngine(
  config: OptiXEngineConfig,
  userSetMaxInstances: Boolean = false
)(using profilingConfig: ProfilingConfig)
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
  private val currentObjectSpecs =
    new AtomicReference[Option[List[ObjectSpec]]](Some(objectSpecs))

  // Keyboard handler for 4D rotation (initialized in finalizeCreate)
  // Using AtomicReference to avoid var
  private val keyHandler = new AtomicReference[Option[OptiXKeyHandler]](None)

  // Track if we have 4D projected objects (need rebuild on rotation)
  private lazy val has4DObjects: Boolean =
    currentObjectSpecs.get().exists(_.exists(spec => ObjectType.isProjected4D(spec.objectType)))

  // Handle rotation/projection events from keyboard and mouse
  override def handleEvent(event: RotationProjectionParameters): Unit =
    logger.debug(s"Received rotation event: rotXW=${event.rotXW}, rotYW=${event.rotYW}, rotZW=${event.rotZW}")
    if has4DObjects then
      // Update object specs with new rotation and projection values
      currentObjectSpecs.set(currentObjectSpecs.get().map(_.map { spec =>
        if ObjectType.isProjected4D(spec.objectType) then
          val currentProj = spec.projection4D.getOrElse(Projection4DSpec.default)
          // Apply eyeW change using same exponential formula as Projection.+
          // Only when event carries an explicit (non-default) eyeW value
          val newEyeW =
            if event.eyeW != Const.defaultEyeW then
              val updatedProjection = Projection(currentProj.eyeW, currentProj.screenW) + event.projection
              updatedProjection.eyeW
            else
              currentProj.eyeW
          val newProj = currentProj.copy(
            eyeW = newEyeW,
            rotXW = currentProj.rotXW + event.rotXW,
            rotYW = currentProj.rotYW + event.rotYW,
            rotZW = currentProj.rotZW + event.rotZW
          )
          logger.debug(s"Updated tesseract: rotXW=${newProj.rotXW}, rotYW=${newProj.rotYW}, rotZW=${newProj.rotZW}, eyeW=${newProj.eyeW}")
          spec.copy(projection4D = Some(newProj))
        else
          spec
      }))
      // Rebuild scene with updated rotation/projection
      rebuildScene()
      // Mark resources as needing render and request it
      renderResources.markNeedsRender()
      GdxRuntime.requestRendering()

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

  // Composition: Three focused components instead of one god object
  private val rendererWrapper = OptiXRendererWrapper(execution.maxInstances)
  private val sceneConfigurator = SceneConfigurator(
    Failure(UnsupportedOperationException("Legacy geometry generator removed. Use objectSpecs instead.")),
    camera.position,
    camera.lookAt,
    camera.up,
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
      GdxRuntime.exit()
    }.get  // Intentional .get - initialization failure should crash (documented)

  private def classifyScene(specs: List[ObjectSpec]): SceneType =
    val types = specs.map(_.objectType.toLowerCase).toSet

    if types.contains("cube-sponge") then
      SceneType.CubeSponges(specs)
    else if types.forall(_ == "sphere") then
      SceneType.Spheres(specs)
    else if types.forall(isTriangleMeshType) then
      SceneType.TriangleMeshes(specs)
    else
      // Mixed scene - spheres + triangle meshes
      val hasSpheres = types.contains("sphere")
      val meshTypes = types.filter(isTriangleMeshType)

      if hasSpheres && meshTypes.size == 1 then
        // Simple mixed: spheres + one mesh type (SUPPORTED)
        SceneType.SimpleMixed(specs, meshTypes.head)
      else if hasSpheres && meshTypes.size > 1 then
        // Check if all mesh types are 4D projected and compatible
        val all4DProjected = meshTypes.forall(ObjectType.isProjected4D)
        if all4DProjected then
          // All 4D objects can share GAS - treat as SimpleMixed with first type
          SceneType.SimpleMixed(specs, meshTypes.head)
        else
          // Complex mixed: spheres + multiple incompatible mesh types (NOT SUPPORTED)
          SceneType.ComplexMixed(specs)
      else
        // Other mixed scenarios
        SceneType.ComplexMixed(specs)

  private def isTriangleMeshType(objectType: String): Boolean =
    objectType == "cube" || ObjectType.isSponge(objectType) || ObjectType.isProjected4D(objectType)

  private def selectSceneBuilder(sceneType: SceneType): Option[SceneBuilder] =
    sceneType match
      case SceneType.Spheres(_) => Some(SphereSceneBuilder())
      case SceneType.TriangleMeshes(specs) =>
        // Check if 4D projected types with edge rendering - use specialized builder
        val all4DProjected = specs.forall(s => ObjectType.isProjected4D(s.objectType))
        val hasEdgeRendering = specs.exists(_.hasEdgeRendering)
        if all4DProjected && hasEdgeRendering then
          Some(TesseractEdgeSceneBuilder(execution.textureDir)(using profilingConfig))
        else
          Some(TriangleMeshSceneBuilder(execution.textureDir)(using profilingConfig))
      case SceneType.CubeSponges(_) => Some(CubeSpongeSceneBuilder())
      case SceneType.SimpleMixed(_, _) => None  // Handled specially in createMultiObjectScene
      case SceneType.ComplexMixed(_) => None

  private def selectMeshBuilder(specs: List[ObjectSpec]): SceneBuilder =
    val firstType = specs.head.objectType.toLowerCase
    val all4DProjected = specs.forall(s => ObjectType.isProjected4D(s.objectType))
    val hasEdgeRendering = specs.exists(_.hasEdgeRendering)

    if all4DProjected && hasEdgeRendering then
      TesseractEdgeSceneBuilder(execution.textureDir)(using profilingConfig)
    else
      TriangleMeshSceneBuilder(execution.textureDir)(using profilingConfig)

  private def createMultiObjectScene(specs: List[ObjectSpec]): Try[Unit] =
    logger.info(s"Creating OptiXEngine with ${specs.length} objects")

    // Warn about high sponge levels before generating geometry
    specs.foreach(warnIfHighLevel)

    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureLights(renderer)
    sceneConfigurator.configurePlanes(renderer, environment.planes)
    sceneConfigurator.configureCamera(renderer)

    // Determine scene type and setup using strategy pattern
    val result = classifyScene(specs) match
      case SceneType.SimpleMixed(specs, meshType) =>
        // Spheres + one triangle mesh type - SUPPORTED
        Try {
          val sphereSpecs = specs.filter(_.objectType.toLowerCase == "sphere")
          val meshSpecs = specs.filterNot(_.objectType.toLowerCase == "sphere")

          logger.info(s"Mixed scene: ${sphereSpecs.size} spheres + ${meshSpecs.size} mesh objects")

          // Calculate effective maxInstances for mesh builder (may need auto-adjustment)
          val meshBuilder = selectMeshBuilder(meshSpecs)
          val effectiveMaxInstances = if !userSetMaxInstances then
            val required = meshBuilder.calculateRequiredInstances(meshSpecs)
            if required > 0 && required > execution.maxInstances then
              val adjusted = Math.min(required * 2, menger.common.Const.maxInstancesLimit)
              logger.info(s"Auto-adjusting max instances: ${execution.maxInstances} → $adjusted (scene requires $required)")
              adjusted
            else
              execution.maxInstances
          else
            execution.maxInstances

          // Build spheres with SphereSceneBuilder
          if sphereSpecs.nonEmpty then
            val sphereBuilder = SphereSceneBuilder()
            sphereBuilder.buildScene(sphereSpecs, renderer, effectiveMaxInstances).get

          // Build triangle meshes with appropriate builder
          if meshSpecs.nonEmpty then
            meshBuilder.buildScene(meshSpecs, renderer, effectiveMaxInstances).get
        }

      case SceneType.ComplexMixed(specs) =>
        val objectTypes = specs.map(_.objectType).distinct
        Failure(UnsupportedOperationException(
          "Cannot mix spheres with multiple different triangle mesh types. " +
          s"Objects: ${objectTypes.mkString(", ")}. " +
          "Spheres can be mixed with one mesh type at a time."
        ))

      case sceneType =>
        selectSceneBuilder(sceneType) match
          case Some(builder) =>
            // Auto-adjust maxInstances if user didn't explicitly set it
            val effectiveMaxInstances = if !userSetMaxInstances then
              val required = builder.calculateRequiredInstances(specs)
              if required > 0 && required > execution.maxInstances then
                val adjusted = Math.min(required * 2, menger.common.Const.maxInstancesLimit)
                logger.info(s"Auto-adjusting max instances: ${execution.maxInstances} → $adjusted (scene requires $required)")
                adjusted
              else
                execution.maxInstances
            else
              execution.maxInstances

            builder.validate(specs, effectiveMaxInstances) match
              case Left(error) => Failure(ValidationException(error, "objectSpecs", specs.map(_.objectType)))
              case Right(_) => builder.buildScene(specs, renderer, effectiveMaxInstances)
          case None =>
            Failure(UnsupportedOperationException(s"No builder available for $sceneType"))

    result.flatMap { _ =>
      Try {
        renderer.setRenderConfig(config.render)
        renderer.setCausticsConfig(config.caustics)
        environment.background.foreach(sceneConfigurator.setBackgroundColor(renderer, _))
        finalizeCreate()
      }
    }

  private def resetTo4DDefaults(): Unit =
    logger.debug("Resetting 4D view to initial state")
    currentObjectSpecs.set(Some(objectSpecs))
    rebuildScene()
    renderResources.markNeedsRender()
    GdxRuntime.requestRendering()

  private def finalizeCreate(): Unit =
    // Register input multiplexer for mouse-based camera control and keyboard shortcuts
    val handler = OptiXKeyHandler(eventDispatcher, onReset = () => resetTo4DDefaults())
    keyHandler.set(Some(handler))
    GdxRuntime.setInputProcessor(OptiXInputMultiplexer(cameraController, handler))

    // Disable continuous rendering - we'll request renders only when needed
    GdxRuntime.setContinuousRendering(false)
    GdxRuntime.requestRendering()

    if execution.timeout > 0 then startExitTimer(execution.timeout)

  private def rebuildScene(): Unit =
    currentObjectSpecs.get() match
      case Some(specs) =>
        Try {
          logger.debug("Rebuilding scene with updated rotation")

          val renderer = rendererWrapper.renderer

          // Save current camera state
          val savedEye = cameraController.currentEye
          val savedLookAt = cameraController.currentLookAt
          val savedUp = cameraController.currentUp
          logger.debug(s"Saving camera state: eye=$savedEye, lookAt=$savedLookAt")

          // Clear geometry without destroying OptiX context
          // This is much lighter weight than dispose/initialize
          renderer.clearAllInstances()

          // Rebuild geometry based on scene type using strategy pattern
          classifyScene(specs) match
            case SceneType.SimpleMixed(specs, meshType) =>
              // Handle mixed scenes specially (spheres + one mesh type)
              val sphereSpecs = specs.filter(_.objectType.toLowerCase == "sphere")
              val meshSpecs = specs.filterNot(_.objectType.toLowerCase == "sphere")

              logger.debug(s"Rebuilding mixed scene: ${sphereSpecs.size} spheres + ${meshSpecs.size} mesh objects")

              // Calculate effective maxInstances for mesh builder
              val meshBuilder = selectMeshBuilder(meshSpecs)
              val effectiveMaxInstances = if !userSetMaxInstances then
                val required = meshBuilder.calculateRequiredInstances(meshSpecs)
                if required > 0 && required > execution.maxInstances then
                  val adjusted = Math.min(required * 2, menger.common.Const.maxInstancesLimit)
                  logger.debug(s"Auto-adjusting max instances for rebuild: ${execution.maxInstances} → $adjusted")
                  adjusted
                else
                  execution.maxInstances
              else
                execution.maxInstances

              // Rebuild spheres
              if sphereSpecs.nonEmpty then
                val sphereBuilder = SphereSceneBuilder()
                sphereBuilder.buildScene(sphereSpecs, renderer, effectiveMaxInstances).get

              // Rebuild meshes
              if meshSpecs.nonEmpty then
                meshBuilder.buildScene(meshSpecs, renderer, effectiveMaxInstances).get

            case SceneType.ComplexMixed(specs) =>
              logger.warn("Cannot rebuild complex mixed scene type")
              Failure(UnsupportedOperationException("Complex mixed scenes not supported for rebuilding")).get

            case sceneType =>
              selectSceneBuilder(sceneType) match
                case Some(builder) =>
                  builder.buildScene(specs, renderer, execution.maxInstances).get
                case None =>
                  logger.warn(s"Cannot rebuild scene type: $sceneType")
                  Failure(UnsupportedOperationException(s"Scene type $sceneType not supported for rebuilding")).get

          // Restore camera to saved position
          cameraState.updateCamera(renderer, savedEye, savedLookAt, savedUp)
          logger.debug(s"Restored camera state: eye=$savedEye, lookAt=$savedLookAt")

          logger.debug("Scene rebuild complete")
        }.recover { case e =>
          logger.error(s"Failed to rebuild scene: ${e.getMessage}", e)
        }
      case None =>
        logger.warn("Cannot rebuild single-object scene interactively")

  override def render(): Unit =
    GdxRuntime.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)

    // Update keyboard handler for 4D rotation (Shift+arrow keys)
    keyHandler.get().foreach(_.update(GdxRuntime.deltaTime))

    val width  = GdxRuntime.width
    val height = GdxRuntime.height

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
      GdxRuntime.exit()

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
