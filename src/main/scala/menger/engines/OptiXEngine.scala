package menger.engines

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.ColorConversions.toCommonColor
import menger.ObjectSpec
import menger.OptiXRenderResources
import menger.PlaneColorSpec
import menger.PlaneSpec
import menger.ProfilingConfig
import menger.common.Const
import menger.common.ImageSize
import menger.input.OptiXCameraController
import menger.input.OptiXInputMultiplexer
import menger.objects.Cube
import menger.objects.CubeSpongeGenerator
import menger.objects.SpongeBySurface
import menger.objects.SpongeByVolume
import menger.optix.CameraState
import menger.optix.CausticsConfig
import menger.optix.OptiXRenderer
import menger.optix.OptiXRendererWrapper
import menger.optix.RenderConfig
import menger.optix.SceneConfigurator

class OptiXEngine(
  val spongeType: String,
  val spongeLevel: Float,
  val lines: Boolean,
  val color: Color,
  val fpsLogIntervalMs: Int,
  val sphereRadius: Float,
  val ior: Float,
  val scale: Float,
  val cameraPos: Vector3,
  val cameraLookat: Vector3,
  val cameraUp: Vector3,
  val center: Vector3,
  val planeSpec: PlaneSpec,
  val planeColor: Option[PlaneColorSpec] = None,
  val timeout: Float = 0f,
  saveName: Option[String] = None,
  val enableStats: Boolean = false,
  val lights: Option[List[menger.LightSpec]] = None,
  val renderConfig: RenderConfig = RenderConfig.Default,
  val causticsConfig: CausticsConfig = CausticsConfig.Disabled,
  val maxInstances: Int = 64,
  val objectSpecs: Option[List[ObjectSpec]] = None
)(using profilingConfig: ProfilingConfig) extends RenderEngine with TimeoutSupport with LazyLogging with SavesScreenshots:

  // Level thresholds for warnings (based on triangle counts and performance)
  private val VolumeLevelWarning = Const.Engine.spongeLevelWarningThreshold
  private val SurfaceLevelWarning = Const.Engine.spongeLevelWarningThreshold
  private val VolumeLevelMax = Const.Engine.cubeSpongeMaxLevel
  private val SurfaceLevelMax = Const.Engine.cubeSpongeMaxLevel

  private def warnIfHighLevel(): Unit =
    val intLevel = spongeLevel.toInt
    spongeType match
      case "sponge-volume" =>
        if intLevel >= VolumeLevelWarning then
          val estimatedTriangles = math.pow(Const.Engine.cubesPerSpongeLevel, intLevel).toLong * Const.Engine.trianglesPerCube
          logger.warn(s"Sponge level $intLevel may be slow (~${estimatedTriangles / 1000}K triangles)")
        if intLevel > VolumeLevelMax then
          logger.error(s"Sponge level $intLevel exceeds recommended maximum ($VolumeLevelMax)")
      case "sponge-surface" =>
        if intLevel >= SurfaceLevelWarning then
          val estimatedTriangles = math.pow(Const.Engine.trianglesPerCube, intLevel).toLong * 6 * 2 // 12^level sub-faces * 6 faces * 2 triangles
          logger.warn(s"Sponge level $intLevel may be slow (~${estimatedTriangles / 1000}K triangles)")
        if intLevel > SurfaceLevelMax then
          logger.error(s"Sponge level $intLevel exceeds recommended maximum ($SurfaceLevelMax)")
      case _ => // No warning for other types

  private val geometryGenerator: Try[OptiXRenderer => Unit] = spongeType match {
    case "sphere" => Try(_.setSphere(menger.common.Vector[3](center.x, center.y, center.z), sphereRadius))
    case "cube" => Try { renderer =>
      val cube = Cube(center = center, scale = sphereRadius * 2)  // Use radius as half-size for consistency
      val mesh = cube.toTriangleMesh
      renderer.setTriangleMesh(mesh)
    }
    case "sponge-volume" => Try { renderer =>
      val sponge = SpongeByVolume(center = center, scale = sphereRadius * 2, level = spongeLevel)
      val mesh = sponge.toTriangleMesh
      renderer.setTriangleMesh(mesh)
    }
    case "sponge-surface" => Try { renderer =>
      given menger.ProfilingConfig = profilingConfig
      val sponge = SpongeBySurface(center = center, scale = sphereRadius * 2, level = spongeLevel)
      val mesh = sponge.toTriangleMesh
      renderer.setTriangleMesh(mesh)
    }
    case _ => Failure(UnsupportedOperationException(spongeType))
  }

  // Composition: Three focused components instead of one god object
  private val rendererWrapper = OptiXRendererWrapper(maxInstances)
  private val sceneConfigurator = SceneConfigurator(geometryGenerator, cameraPos, cameraLookat, cameraUp, planeSpec, lights)
  private val cameraState = CameraState(cameraPos, cameraLookat, cameraUp)

  private val renderResources: OptiXRenderResources = OptiXRenderResources(0, 0)
  private lazy val cameraController: OptiXCameraController =
    OptiXCameraController(rendererWrapper, cameraState, renderResources, cameraPos, cameraLookat, cameraUp)

  override def create(): Unit =
    val result = objectSpecs match
      case Some(specs) if specs.nonEmpty =>
        createMultiObjectScene(specs)
      case _ =>
        createSingleObjectScene()

    result.recover { case e: Exception =>
      logger.error(s"Failed to create OptiX scene: ${e.getMessage}", e)
      Gdx.app.exit()
    }.get  // Intentional .get - initialization failure should crash (documented)

  private def createSingleObjectScene(): Try[Unit] = Try:
    logger.info(s"Creating OptiXEngine with object=$spongeType, radius=$sphereRadius, color=$color, ior=$ior, scale=$scale, renderConfig=$renderConfig, causticsConfig=$causticsConfig")

    // Warn about high sponge levels before generating geometry
    warnIfHighLevel()

    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureScene(renderer)

    // Configure color and IOR based on object type
    spongeType match
      case "sphere" =>
        sceneConfigurator.setSphereColor(renderer, color.toCommonColor)
        sceneConfigurator.setIOR(renderer, ior)
      case "cube" | "sponge-volume" | "sponge-surface" =>
        sceneConfigurator.setTriangleMeshColor(renderer, color.toCommonColor)
        sceneConfigurator.setTriangleMeshIOR(renderer, ior)
      case _ =>
        // For other types, try both (backward compatibility)
        sceneConfigurator.setSphereColor(renderer, color.toCommonColor)
        sceneConfigurator.setIOR(renderer, ior)

    sceneConfigurator.setScale(renderer, scale)
    renderer.setRenderConfig(renderConfig)
    renderer.setCausticsConfig(causticsConfig)
    planeColor.foreach(sceneConfigurator.setPlaneColor(renderer, _))

    finalizeCreate()

  private def createMultiObjectScene(specs: List[ObjectSpec]): Try[Unit] = Try:
    logger.info(s"Creating OptiXEngine with ${specs.length} objects")

    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureLights(renderer)
    sceneConfigurator.configurePlane(renderer)
    sceneConfigurator.configureCamera(renderer)

    // Validate that all objects are compatible with IAS
    validateObjectSpecs(specs) match
      case Left(error) => Failure(IllegalArgumentException(error))
      case Right(_) =>
        // Determine scene type and setup
        val objectTypes = specs.map(_.objectType).distinct
        val setupResult = if objectTypes.contains("cube-sponge") then
          // cube-sponge generates many instances from one spec - handle specially
          setupCubeSponges(specs, renderer)
        else if objectTypes.forall(_ == "sphere") then
          setupMultipleSpheres(specs, renderer)
        else if objectTypes.forall(t => t == "cube" || t == "sponge-volume" || t == "sponge-surface") then
          setupMultipleTriangleMeshes(specs, renderer)
        else
          Failure(UnsupportedOperationException(
            "Cannot mix spheres and triangle meshes in the same scene yet. " +
            s"Objects: ${objectTypes.mkString(", ")}"
          ))

        setupResult.get  // Propagate failure
        ()  // Return Unit for Try[Unit]

    renderer.setRenderConfig(renderConfig)
    renderer.setCausticsConfig(causticsConfig)
    planeColor.foreach(sceneConfigurator.setPlaneColor(renderer, _))

    finalizeCreate()

  private def validateObjectSpecs(specs: List[ObjectSpec]): Either[String, Unit] =
    if specs.isEmpty then
      Left("Object specs list cannot be empty")
    else if specs.length > maxInstances then
      Left(s"Too many objects: ${specs.length} exceeds max instances limit of $maxInstances. " +
        "Use --max-instances to increase the limit.")
    else
      Right(())

  private def setupMultipleSpheres(specs: List[ObjectSpec], renderer: OptiXRenderer): Try[Unit] = Try:
    logger.info(s"Setting up ${specs.length} sphere instances")

    // addSphereInstance() automatically enables IAS mode - do NOT call setSphere() first!
    specs.foreach { spec =>
      val color = spec.color.getOrElse(menger.common.Color(0.7f, 0.7f, 0.7f))
      val scale = spec.size

      // Create 4x3 transform matrix with scale and translation
      // Format: 3 rows x 4 columns, row-major storage
      // [row0: sx, 0, 0, tx], [row1: 0, sy, 0, ty], [row2: 0, 0, sz, tz]
      val transform = Array(
        scale, 0f, 0f, spec.x,
        0f, scale, 0f, spec.y,
        0f, 0f, scale, spec.z
      )

      val instanceId = renderer.addSphereInstance(transform, color, spec.ior)

      instanceId match
        case Some(id) =>
          logger.debug(s"Added sphere instance $id at position=(${spec.x}, ${spec.y}, ${spec.z}), scale=$scale, color=$color, ior=${spec.ior}")
        case None =>
          logger.error(s"Failed to add sphere instance at position=(${spec.x}, ${spec.y}, ${spec.z})")
    }

  private def setupMultipleTriangleMeshes(specs: List[ObjectSpec], renderer: OptiXRenderer): Try[Unit] = Try:
    logger.info(s"Setting up ${specs.length} triangle mesh instances")

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

    // Add instances
    specs.foreach { spec =>
      val position = menger.common.Vector[3](spec.x, spec.y, spec.z)
      val color = spec.color.getOrElse(menger.common.Color(0.7f, 0.7f, 0.7f))
      val instanceId = renderer.addTriangleMeshInstance(position, color, spec.ior)

      instanceId match
        case Some(id) =>
          logger.debug(s"Added ${spec.objectType} instance $id at position=(${spec.x}, ${spec.y}, ${spec.z}), color=$color, ior=${spec.ior}")
        case None =>
          logger.error(s"Failed to add ${spec.objectType} instance at position=(${spec.x}, ${spec.y}, ${spec.z})")
    }

  private def createMeshForSpec(spec: ObjectSpec): menger.common.TriangleMeshData =
    spec.objectType match
      case "cube" =>
        val cube = Cube(center = Vector3(0f, 0f, 0f), scale = spec.size)
        cube.toTriangleMesh
      case "sponge-volume" =>
        require(spec.level.isDefined, "sponge-volume requires level")
        val sponge = SpongeByVolume(center = Vector3(0f, 0f, 0f), scale = spec.size, level = spec.level.get)
        sponge.toTriangleMesh
      case "sponge-surface" =>
        given menger.ProfilingConfig = profilingConfig
        require(spec.level.isDefined, "sponge-surface requires level")
        val sponge = SpongeBySurface(center = Vector3(0f, 0f, 0f), scale = spec.size, level = spec.level.get)
        sponge.toTriangleMesh
      case other =>
        require(false, s"Unknown mesh type: $other")
        ???  // Never reached due to require, but needed for type checker

  private def isCompatibleMesh(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    (spec1.objectType, spec2.objectType) match
      case (t1, t2) if t1 == t2 =>
        // Same type - check if levels match for sponges
        (t1, spec1.level, spec2.level) match
          case ("sponge-volume" | "sponge-surface", Some(l1), Some(l2)) => l1 == l2
          case ("sponge-volume" | "sponge-surface", _, _) => false  // Missing level
          case _ => true  // Non-sponge types are always compatible with same type
      case _ => false  // Different types

  private def setupCubeSponges(specs: List[ObjectSpec], renderer: OptiXRenderer): Try[Unit] = {
    // cube-sponge generates many cube instances from each spec
    // Validate that we don't exceed max instances limit
    val totalInstances = specs.map { spec =>
      require(spec.level.isDefined, "cube-sponge requires level")
      val level = spec.level.get.toInt
      Math.pow(Const.Engine.cubesPerSpongeLevel, level).toLong
    }.sum

    if totalInstances > maxInstances then
      Failure(IllegalArgumentException(
        s"cube-sponge specs generate $totalInstances total instances, " +
        s"exceeding max instances limit of $maxInstances. " +
        "Reduce sponge levels or use --max-instances to increase the limit."
      ))
    else Try:
      logger.info(s"Setting up ${specs.length} cube-sponge(s) generating $totalInstances total cube instances")

      // Create base cube mesh (shared by all instances)
      // Level 1 Menger sponge geometry
      val baseCube = Cube(center = Vector3(0f, 0f, 0f), scale = 1.0f)
      renderer.setTriangleMesh(baseCube.toTriangleMesh)

      // For each cube-sponge spec, generate and add all cube instances
      specs.foreach { spec =>
        require(spec.level.isDefined, "cube-sponge requires level")
        val level = spec.level.get.toInt
        val color = spec.color.getOrElse(menger.common.Color(0.7f, 0.7f, 0.7f))

        // Generate all cube transforms using CubeSpongeGenerator
        val generator = CubeSpongeGenerator(
          center = Vector3(spec.x, spec.y, spec.z),
          size = spec.size,
          level = level
        )

        logger.info(s"Generating ${generator.cubeCount} cube instances for level $level cube-sponge at (${spec.x}, ${spec.y}, ${spec.z})")

        // Add each cube as an instance
        generator.generateTransforms.foreach { case (position, scale) =>
          val posVec = menger.common.Vector[3](position.x, position.y, position.z)

          // Create 4x3 transform matrix with scale and translation
          val transform = Array(
            scale, 0f, 0f, position.x,
            0f, scale, 0f, position.y,
            0f, 0f, scale, position.z
          )

          val instanceId = renderer.addTriangleMeshInstance(transform, color, spec.ior)

          instanceId match
            case None =>
              logger.error(s"Failed to add cube instance at position=($position), scale=$scale")
            case Some(_) =>
              // Success - don't log each instance (too verbose for 8000+ cubes)
        }

        logger.debug(s"Added ${generator.cubeCount} cube instances for cube-sponge")
      }
    
  }

  private def finalizeCreate(): Unit =
    // Register input multiplexer for mouse-based camera control and keyboard shortcuts
    Gdx.input.setInputProcessor(OptiXInputMultiplexer(cameraController))

    // Disable continuous rendering - we'll request renders only when needed
    Gdx.graphics.setContinuousRendering(false)
    Gdx.graphics.requestRendering()

    if timeout > 0 then startExitTimer(timeout)

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
        val rgbaBytes = if enableStats then renderWithStats(width, height) else rendererWrapper.renderScene(size)
        renderResources.renderToScreen(rgbaBytes, width, height)
      else
        // Just redraw the existing texture without re-rendering
        renderResources.redrawExisting(width, height)

      saveImage()

    // Exit after saving when in non-interactive mode (unless timeout is set)
    if saveName.isDefined && !renderResources.hasSaved && timeout == 0 then
      renderResources.markSaved()
      Gdx.app.exit()

  protected def currentSaveName: Option[String] = saveName

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
