package menger.engines

import scala.util.Failure
import scala.util.Success
import scala.util.Try

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.ColorConversions.toCommonColor
import menger.ObjectSpec
import menger.OptiXRenderResources
import menger.ProfilingConfig
import menger.Projection4DSpec
import menger.RotationProjectionParameters
import menger.TextureLoader
import menger.Vector3Extensions.toVector3
import menger.common.ConfigurationException
import menger.common.Const
import menger.common.ImageSize
import menger.common.ObjectType
import menger.common.TransformUtil
import menger.common.ValidationException
import menger.config.OptiXEngineConfig
import menger.input.EventDispatcher
import menger.input.Observer
import menger.input.OptiXCameraHandler
import menger.input.OptiXInputMultiplexer
import menger.input.OptiXKeyHandler
import menger.objects.Cube
import menger.objects.CubeSpongeGenerator
import menger.objects.SpongeBySurface
import menger.objects.SpongeByVolume
import menger.objects.higher_d.TesseractMesh
import menger.optix.CameraState
import menger.optix.Material
import menger.optix.OptiXRenderer
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

  // Public accessors for testing
  def sphereRadius: Float = scene.sphereRadius

  // Event dispatcher for 4D rotation events
  private val eventDispatcher = EventDispatcher().withObserver(this)

  // Mutable state for current object specs (for interactive rotation updates)
  // Wartremover: var required for interactive rotation state management
  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var currentObjectSpecs: Option[List[ObjectSpec]] = scene.objectSpecs

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

  private def warnIfHighLevel(): Unit =
    val intLevel = scene.level.toInt
    scene.spongeType match
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

  private val geometryGenerator: Try[OptiXRenderer => Unit] = scene.spongeType match {
    case "sphere" => Try(_.setSphere(scene.center.toVector3, scene.sphereRadius))
    case "cube" => Try { renderer =>
      val cube = Cube(center = scene.center, scale = scene.sphereRadius * 2)
      val mesh = cube.toTriangleMesh
      renderer.setTriangleMesh(mesh)
    }
    case "sponge-volume" => Try { renderer =>
      val sponge = SpongeByVolume(center = scene.center, scale = scene.sphereRadius * 2, level = scene.level)
      val mesh = sponge.toTriangleMesh
      renderer.setTriangleMesh(mesh)
    }
    case "sponge-surface" => Try { renderer =>
      given menger.ProfilingConfig = profilingConfig
      val sponge = SpongeBySurface(center = scene.center, scale = scene.sphereRadius * 2, level = scene.level)
      val mesh = sponge.toTriangleMesh
      renderer.setTriangleMesh(mesh)
    }
    case _ => Failure(UnsupportedOperationException(scene.spongeType))
  }

  // Composition: Three focused components instead of one god object
  private val rendererWrapper = OptiXRendererWrapper(execution.maxInstances)
  private val sceneConfigurator = SceneConfigurator(
    geometryGenerator,
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
    val result = scene.objectSpecs match
      case Some(specs) if specs.nonEmpty =>
        createMultiObjectScene(specs)
      case _ =>
        createSingleObjectScene()

    result.recover { case e: Exception =>
      logger.error(s"Failed to create OptiX scene: ${e.getMessage}", e)
      Gdx.app.exit()
    }.get  // Intentional .get - initialization failure should crash (documented)

  private def createSingleObjectScene(): Try[Unit] = Try:
    val color = scene.material.color
    val ior = scene.material.ior
    logger.info(s"Creating OptiXEngine with object=${scene.spongeType}, radius=${scene.sphereRadius}, color=$color, ior=$ior, scale=${scene.scale}, renderConfig=${config.render}, causticsConfig=${config.caustics}")

    // Warn about high sponge levels before generating geometry
    warnIfHighLevel()

    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureScene(renderer)

    // Configure color and IOR based on object type
    scene.spongeType match
      case "sphere" =>
        sceneConfigurator.setSphereColor(renderer, color.toCommonColor)
        sceneConfigurator.setIOR(renderer, ior)
      case _ if ObjectType.isSpongeOrCube(scene.spongeType) =>
        sceneConfigurator.setTriangleMeshColor(renderer, color.toCommonColor)
        sceneConfigurator.setTriangleMeshIOR(renderer, ior)
      case _ =>
        // For other types, try both (backward compatibility)
        sceneConfigurator.setSphereColor(renderer, color.toCommonColor)
        sceneConfigurator.setIOR(renderer, ior)

    sceneConfigurator.setScale(renderer, scene.scale)
    renderer.setRenderConfig(config.render)
    renderer.setCausticsConfig(config.caustics)
    environment.planeColor.foreach(sceneConfigurator.setPlaneColor(renderer, _))

    finalizeCreate()

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

  private def createMultiObjectScene(specs: List[ObjectSpec]): Try[Unit] =
    logger.info(s"Creating OptiXEngine with ${specs.length} objects")

    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureLights(renderer)
    sceneConfigurator.configurePlane(renderer)
    sceneConfigurator.configureCamera(renderer)

    // Validate that all objects are compatible with IAS
    val validationResult = validateObjectSpecs(specs) match
      case Left(error) => Failure(ValidationException(error, "objectSpecs", specs.map(_.objectType)))
      case Right(_) =>
        // Determine scene type and setup using pattern matching
        classifyScene(specs) match
          case SceneType.CubeSponges(specs) =>
            // cube-sponge generates many instances from one spec - handle specially
            setupCubeSponges(specs, renderer)
          case SceneType.Spheres(specs) =>
            setupMultipleSpheres(specs, renderer)
          case SceneType.TriangleMeshes(specs) =>
            setupMultipleTriangleMeshes(specs, renderer)
          case SceneType.Mixed(specs) =>
            val objectTypes = specs.map(_.objectType).distinct
            Failure(UnsupportedOperationException(
              "Cannot mix spheres and triangle meshes in the same scene yet. " +
              s"Objects: ${objectTypes.mkString(", ")}"
            ))

    validationResult.flatMap { _ =>
      Try {
        renderer.setRenderConfig(config.render)
        renderer.setCausticsConfig(config.caustics)
        environment.planeColor.foreach(sceneConfigurator.setPlaneColor(renderer, _))
        finalizeCreate()
      }
    }

  private def validateObjectSpecs(specs: List[ObjectSpec]): Either[String, Unit] =
    if specs.isEmpty then
      Left("Object specs list cannot be empty")
    else if specs.length > execution.maxInstances then
      Left(s"Too many objects: ${specs.length} exceeds max instances limit of ${execution.maxInstances}. " +
        "Use --max-instances to increase the limit.")
    else
      Right(())

  private val defaultColor = menger.common.Color(0.7f, 0.7f, 0.7f)

  private def extractMaterial(spec: ObjectSpec): Material =
    spec.material match
      case Some(mat) => mat
      case None => Material(spec.color.getOrElse(defaultColor), spec.ior)

  private def setupMultipleSpheres(specs: List[ObjectSpec], renderer: OptiXRenderer): Try[Unit] = Try:
    logger.debug(s"Setting up ${specs.length} sphere instances")

    // addSphereInstance() automatically enables IAS mode - do NOT call setSphere() first!
    specs.foreach { spec =>
      val material = extractMaterial(spec)
      val scale = spec.size

      val transform = TransformUtil.createScaleTranslation(scale, spec.x, spec.y, spec.z)

      val instanceId = renderer.addSphereInstance(transform, material)

      instanceId match
        case Some(id) =>
          logger.debug(s"Added sphere instance $id at position=(${spec.x}, ${spec.y}, ${spec.z}), scale=$scale, material=$material")
        case None =>
          logger.error(s"Failed to add sphere instance at position=(${spec.x}, ${spec.y}, ${spec.z})")
    }

  private def setupMultipleTriangleMeshes(specs: List[ObjectSpec], renderer: OptiXRenderer): Try[Unit] = Try:
    logger.debug(s"Setting up ${specs.length} triangle mesh instances")

    // For triangle meshes, we need all specs to be compatible (same geometry type)
    // We'll use the first spec to determine the mesh type
    val firstSpec = specs.head
    val mesh = createMeshForSpec(firstSpec)
    renderer.setTriangleMesh(mesh)

    // Validate that all specs use compatible geometry
    specs.foreach { spec =>
      require(isCompatibleMesh(spec, firstSpec),
        s"Cannot mix different triangle mesh types. First object is ${firstSpec.objectType}, " +
        s"but found ${spec.objectType}. All triangle mesh objects must be the same type for now."
      )
    }

    // Load textures and build a map from filename to texture index
    val textureIndices = loadTexturesForSpecs(specs, renderer)

    // Add instances
    specs.foreach { spec =>
      val position = menger.common.Vector[3](spec.x, spec.y, spec.z)
      val material = extractMaterial(spec)

      // Get texture index if this spec has a texture
      val textureIndex = spec.texture.flatMap(textureIndices.get).getOrElse(-1)

      val instanceId = renderer.addTriangleMeshInstance(position, material, textureIndex)

      instanceId match
        case Some(id) =>
          val textureInfo = if textureIndex >= 0 then s", texture=$textureIndex" else ""
          logger.debug(s"Added ${spec.objectType} instance $id at position=(${spec.x}, ${spec.y}, ${spec.z}), material=$material$textureInfo")
        case None =>
          logger.error(s"Failed to add ${spec.objectType} instance at position=(${spec.x}, ${spec.y}, ${spec.z})")
    }

  private def loadTexturesForSpecs(
    specs: List[ObjectSpec],
    renderer: OptiXRenderer
  ): Map[String, Int] =
    // Collect unique texture filenames
    val textureFilenames = specs.flatMap(_.texture).distinct

    if textureFilenames.isEmpty then
      Map.empty
    else
      logger.info(s"Loading ${textureFilenames.length} texture(s)")

      textureFilenames.flatMap { filename =>
        TextureLoader.load(filename, execution.textureDir) match
          case scala.util.Success(textureData) =>
            renderer.uploadTexture(textureData.name, textureData.data, textureData.width, textureData.height) match
              case scala.util.Success(index) =>
                logger.debug(s"Uploaded texture '$filename' as index $index")
                Some(filename -> index)
              case scala.util.Failure(e) =>
                logger.error(s"Failed to upload texture '$filename': ${e.getMessage}")
                None
          case scala.util.Failure(e) =>
            logger.error(s"Failed to load texture '$filename': ${e.getMessage}")
            None
      }.toMap

  private def createMeshForSpec(spec: ObjectSpec): menger.common.TriangleMeshData =
    spec.objectType match
      case "cube" =>
        val cube = Cube(center = Vector3(0f, 0f, 0f), scale = spec.size)
        cube.toTriangleMesh
      case "sponge-volume" =>
        require(spec.level.isDefined, "sponge-volume requires level")
        // Safe .get: level validated by require above
        val sponge = SpongeByVolume(center = Vector3(0f, 0f, 0f), scale = spec.size, level = spec.level.get)
        sponge.toTriangleMesh
      case "sponge-surface" =>
        given menger.ProfilingConfig = profilingConfig
        require(spec.level.isDefined, "sponge-surface requires level")
        // Safe .get: level validated by require above
        val sponge = SpongeBySurface(center = Vector3(0f, 0f, 0f), scale = spec.size, level = spec.level.get)
        sponge.toTriangleMesh
      case "tesseract" =>
        // Safe .getOrElse: projection4D should always be Some for tesseract (parser guarantees this)
        val proj = spec.projection4D.getOrElse(Projection4DSpec.default)
        val tesseract = TesseractMesh(
          center = Vector3(0f, 0f, 0f),
          size = spec.size,
          eyeW = proj.eyeW,
          screenW = proj.screenW,
          rotXW = proj.rotXW,
          rotYW = proj.rotYW,
          rotZW = proj.rotZW
        )
        tesseract.toTriangleMesh
      case other =>
        require(false, s"Unknown mesh type: $other")
        ???  // Never reached due to require, but needed for type checker

  private def isCompatibleMesh(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    (spec1.objectType, spec2.objectType) match
      case (t1, t2) if t1 == t2 =>
        // Same type - check if parameters match
        if ObjectType.isSponge(t1) then
          (spec1.level, spec2.level) match
            case (Some(l1), Some(l2)) => l1 == l2
            case _ => false  // Missing level
        else if ObjectType.isHypercube(t1) then
          // Hypercube types are compatible only if 4D projection params match
          (spec1.projection4D, spec2.projection4D) match
            case (Some(p1), Some(p2)) =>
              p1.eyeW == p2.eyeW && p1.screenW == p2.screenW &&
              p1.rotXW == p2.rotXW && p1.rotYW == p2.rotYW && p1.rotZW == p2.rotZW
            case (None, None) => true  // Both using defaults
            case _ => false
        else
          true  // Non-sponge, non-hypercube types are always compatible with same type
      case _ => false  // Different types

  private def calculateInstanceCount(spec: ObjectSpec): Long =
    require(spec.level.isDefined, "cube-sponge requires level")
    // Safe .get: level validated by require above
    val level = spec.level.get.toInt
    Math.pow(Const.Engine.cubesPerSpongeLevel, level).toLong

  private def validateInstanceLimit(specs: List[ObjectSpec]): Try[Unit] =
    val totalInstances = specs.map(calculateInstanceCount).sum
    
    if totalInstances > execution.maxInstances then
      Failure(ConfigurationException(
        s"cube-sponge specs generate $totalInstances total instances, " +
        s"exceeding max instances limit of ${execution.maxInstances}. " +
        "Reduce sponge levels or use --max-instances to increase the limit.",
        Some("maxInstances")
      ))
    else
      Success(())

  private def setupBaseCubeMesh(renderer: OptiXRenderer): Try[Unit] = Try:
    // Create base cube mesh (shared by all instances)
    val baseCube = Cube(center = Vector3(0f, 0f, 0f), scale = 1.0f)
    renderer.setTriangleMesh(baseCube.toTriangleMesh)

  private def addAllCubeInstances(
    specs: List[ObjectSpec],
    renderer: OptiXRenderer
  ): Unit =
    specs.foreach { spec =>
      addCubeInstancesForSpec(spec, renderer)
    }

  private def addCubeInstancesForSpec(
    spec: ObjectSpec,
    renderer: OptiXRenderer
  ): Unit =
    require(spec.level.isDefined, "cube-sponge requires level")
    // Safe .get: level validated by require above
    val level = spec.level.get.toInt
    val material = extractMaterial(spec)

    // Generate all cube transforms using CubeSpongeGenerator
    val generator = CubeSpongeGenerator(
      center = Vector3(spec.x, spec.y, spec.z),
      size = spec.size,
      level = level
    )

    logger.debug(s"Generating ${generator.cubeCount} cube instances for level $level cube-sponge at (${spec.x}, ${spec.y}, ${spec.z})")

    // Add each cube as an instance
    generator.generateTransforms.foreach { case (position, scale) =>
      addSingleCubeInstance(position, scale, material, renderer)
    }

    logger.debug(s"Added ${generator.cubeCount} cube instances for cube-sponge")

  private def addSingleCubeInstance(
    position: Vector3,
    scale: Float,
    material: Material,
    renderer: OptiXRenderer
  ): Unit =
    val transform = TransformUtil.createScaleTranslation(
      scale, position.x, position.y, position.z
    )

    val instanceId = renderer.addTriangleMeshInstance(transform, material)

    instanceId match
      case None =>
        logger.error(s"Failed to add cube instance at position=($position), scale=$scale")
      case Some(_) =>
        // Success - don't log each instance (too verbose for 8000+ cubes)

  private def setupCubeSponges(specs: List[ObjectSpec], renderer: OptiXRenderer): Try[Unit] =
    val totalInstances = specs.map(calculateInstanceCount).sum
    logger.debug(s"Setting up ${specs.length} cube-sponge(s) generating $totalInstances total cube instances")

    for
      _ <- validateInstanceLimit(specs)
      _ <- setupBaseCubeMesh(renderer)
    yield
      addAllCubeInstances(specs, renderer)
  

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

          // Rebuild geometry based on scene type
          classifyScene(specs) match
            case SceneType.TriangleMeshes(meshSpecs) =>
              setupMultipleTriangleMeshes(meshSpecs, renderer).get
            case SceneType.Spheres(sphereSpecs) =>
              setupMultipleSpheres(sphereSpecs, renderer).get
            case SceneType.CubeSponges(cubeSpecs) =>
              setupCubeSponges(cubeSpecs, renderer).get
            case SceneType.Mixed(mixedSpecs) =>
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
