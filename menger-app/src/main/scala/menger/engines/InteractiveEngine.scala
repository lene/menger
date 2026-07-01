package menger.engines

import java.nio.file.Files
import java.nio.file.Paths
import java.util.concurrent.atomic.AtomicReference

import scala.collection.mutable.ArrayBuffer
import scala.util.Try

import com.badlogic.gdx.graphics.GL20
import com.typesafe.scalalogging.LazyLogging
import io.github.lene.optix.CameraState
import io.github.lene.optix.RenderResult
import io.github.lene.optix.SceneConfigurator
import io.github.lene.optix.TextureUploadException
import menger.ObjectSpec
import menger.Projection4DSpec
import menger.RotationProjectionParameters
import menger.Vector3Extensions.toVector3
import menger.common.Const
import menger.common.ImageSize
import menger.common.ObjectType
import menger.common.ProfilingConfig
import menger.config.LevelConfig
import menger.config.OptiXEngineConfig
import menger.dsl.DenoiseMode
import menger.engines.scene.Hexadecachoron4DSceneBuilder
import menger.engines.scene.InstanceId
import menger.engines.scene.Menger4DSceneBuilder
import menger.engines.scene.SceneBuilder
import menger.engines.scene.Sierpinski4DSceneBuilder
import menger.engines.scene.TextureManager
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

  override protected def renderConfig: menger.common.RenderConfig = config.render
  override protected def denoiseMode: DenoiseMode = config.denoiseMode
  override protected def accumulationFrames: Int = config.accumulationFrames

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

  // Coordinate cross visibility (togglable via 'C' key)
  private val crossVisible = new java.util.concurrent.atomic.AtomicBoolean(config.cross.enabled)

  private val CrossConeLength  = 0.2f
  private val CrossConeRadius  = 0.05f
  private val CrossAxisColors  = Map(
    0 -> menger.common.Material(menger.common.Color(1, 0, 0), metallic = 1.0f, roughness = 0.3f),
    1 -> menger.common.Material(menger.common.Color(0, 1, 0), metallic = 1.0f, roughness = 0.3f),
    2 -> menger.common.Material(menger.common.Color(0, 0, 1), metallic = 1.0f, roughness = 0.3f)
  )

  // Event dispatcher for 4D rotation events
  private val eventDispatcher = EventDispatcher().withObserver(this)

  private val currentObjectSpecs =
    new AtomicReference[Option[List[ObjectSpec]]](Some(objectSpecs))

  // Keyboard handler for 4D rotation (initialized in finalizeCreate)
  private val keyHandler = new AtomicReference[Option[OptiXKeyHandler]](None)

  // Track if we have 4D objects (projected triangle mesh OR menger4d OR sierpinski4d OR hexadecachoron4d) that need rebuild on rotation
  private lazy val has4DObjects: Boolean =
    currentObjectSpecs.get().exists(_.exists(spec =>
      ObjectType.isProjected4D(spec.objectType) || ObjectType.isMenger4D(spec.objectType) ||
      ObjectType.isSierpinski4D(spec.objectType) || ObjectType.isHexadecachoron4D(spec.objectType)
    ))

  /** Per-spec instanceId mapping for the menger4d rotation fast path. */
  private case class Menger4DState(
    specs: List[ObjectSpec],
    instancesPerSpec: Vector[Vector[InstanceId]]
  )

  /** Per-spec instanceId mapping for the sierpinski4d rotation fast path. */
  private case class Sierpinski4DState(
    specs: List[ObjectSpec],
    instancesPerSpec: Vector[Vector[InstanceId]]
  )

  /** Per-spec instanceId mapping for the hexadecachoron4d rotation fast path. */
  private case class Hexadecachoron4DState(
    specs: List[ObjectSpec],
    instancesPerSpec: Vector[Vector[InstanceId]]
  )

  /** Cached slot/instance indices for 4D-rotation fast paths.
    * Exactly one variant is active at a time; Empty when no fast path is available. */
  private enum Scene4DCache:
    case Empty
    case Gpu(state: WithAnimation.Anim4DState)
    case Menger4D(state: Menger4DState)
    case Sierpinski4D(state: Sierpinski4DState)
    case Hexadecachoron4D(state: Hexadecachoron4DState)
  // AtomicReference for cross-thread visibility only. All reads and writes happen on the
  // LibGDX GL thread (render() and key handlers are both dispatched there), so the
  // non-atomic get+set compound operations in tryXxx4DFastPath are safe — do not
  // add a second thread that writes this without converting to synchronized or CAS loops.
  private val scene4DCache: AtomicReference[Scene4DCache] =
    new AtomicReference(Scene4DCache.Empty)

  override protected val sceneConfigurator: SceneConfigurator = SceneConfigurator(
    camera.position.toVector3,
    camera.lookAt.toVector3,
    camera.up.toVector3,
    environment.lights.toArray
  )

  override protected val cameraState: CameraState =
    CameraState(camera.position.toVector3, camera.lookAt.toVector3, camera.up.toVector3)

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
        if ObjectType.isProjected4D(spec.objectType) || ObjectType.isMenger4D(spec.objectType) ||
           ObjectType.isSierpinski4D(spec.objectType) || ObjectType.isHexadecachoron4D(spec.objectType) then
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
            s"Updated 4D object: rotXW=${newProj.rotXW}, rotYW=${newProj.rotYW}, " +
            s"rotZW=${newProj.rotZW}, eyeW=${newProj.eyeW}"
          )
          spec.copy(projection4D = Some(newProj))
        else
          spec
      })
      currentObjectSpecs.set(updatedSpecs)

      val fastPathTaken = updatedSpecs.exists(specs =>
        tryRotation4DFastPath(specs, rendererWrapper.renderer) ||
        tryMenger4DFastPath(specs, rendererWrapper.renderer) ||
        trySierpinski4DFastPath(specs, rendererWrapper.renderer) ||
        tryHexadecachoron4DFastPath(specs, rendererWrapper.renderer)
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
    renderer: io.github.lene.optix.OptiXRenderer
  ): Boolean =
    scene4DCache.get match
      case Scene4DCache.Gpu(prev) =>
        val took = RotationFastPath.tryGpuProjectionFastPath(
          newSpecs, renderer, prev.specs, prev.slotsPerSpec
        )
        if took then scene4DCache.set(Scene4DCache.Gpu(prev.copy(specs = newSpecs)))
        took
      case _ => false

  /** Menger4D fast path: update projection params on each recorded instance directly.
    * Returns true iff the menger4d slot map is populated and the change is purely
    * a Projection4DSpec delta — skipping the full geometry rebuild. */
  private def tryMenger4DFastPath(
    newSpecs: List[ObjectSpec],
    renderer: io.github.lene.optix.OptiXRenderer
  ): Boolean =
    scene4DCache.get match
      case Scene4DCache.Menger4D(prev) =>
        val took = RotationFastPath.tryFastPath(
          newSpecs, renderer, prev.specs, prev.instancesPerSpec,
          RotationFastPath.menger4DUpdater
        )
        if took then scene4DCache.set(Scene4DCache.Menger4D(prev.copy(specs = newSpecs)))
        took
      case _ => false

  private def trySierpinski4DFastPath(
    newSpecs: List[ObjectSpec],
    renderer: io.github.lene.optix.OptiXRenderer
  ): Boolean =
    scene4DCache.get match
      case Scene4DCache.Sierpinski4D(prev) =>
        val took = RotationFastPath.tryFastPath(
          newSpecs, renderer, prev.specs, prev.instancesPerSpec,
          RotationFastPath.sierpinski4DUpdater
        )
        if took then scene4DCache.set(Scene4DCache.Sierpinski4D(prev.copy(specs = newSpecs)))
        took
      case _ => false

  private def tryHexadecachoron4DFastPath(
    newSpecs: List[ObjectSpec],
    renderer: io.github.lene.optix.OptiXRenderer
  ): Boolean =
    scene4DCache.get match
      case Scene4DCache.Hexadecachoron4D(prev) =>
        val took = RotationFastPath.tryFastPath(
          newSpecs, renderer, prev.specs, prev.instancesPerSpec,
          RotationFastPath.hexadecachoron4DUpdater
        )
        if took then scene4DCache.set(Scene4DCache.Hexadecachoron4D(prev.copy(specs = newSpecs)))
        took
      case _ => false

  private val levelConfigs: Map[String, LevelConfig] = Map(
    "sponge-volume"      -> LevelConfig(
      Const.Engine.spongeLevelWarningThreshold, Const.Engine.cubeSpongeMaxLevel,
      lvl => math.pow(Const.Engine.cubesPerSpongeLevel, lvl).toLong * Const.Engine.trianglesPerCube),
    "sponge-surface"     -> LevelConfig(
      Const.Engine.spongeLevelWarningThreshold, Const.Engine.cubeSpongeMaxLevel,
      lvl => math.pow(Const.Engine.trianglesPerCube, lvl).toLong * 6 * 2),
    "tesseract-sponge"        -> LevelConfig(
      Const.Engine.tesseractSpongeWarnLevel, Const.Engine.tesseractSpongeMaxLevel,
      TesseractSpongeMesh.estimatedTriangles),
    "tesseract-sponge-volume" -> LevelConfig(
      Const.Engine.tesseractSpongeWarnLevel, Const.Engine.tesseractSpongeMaxLevel,
      TesseractSpongeMesh.estimatedTriangles),
    "tesseract-sponge-2"       -> LevelConfig(
      Const.Engine.tesseractSponge2WarnLevel, Const.Engine.tesseractSponge2MaxLevel,
      TesseractSponge2Mesh.estimatedTriangles),
    "tesseract-sponge-surface" -> LevelConfig(
      Const.Engine.tesseractSponge2WarnLevel, Const.Engine.tesseractSponge2MaxLevel,
      TesseractSponge2Mesh.estimatedTriangles),
  )

  private def warnIfHighLevel(spec: ObjectSpec): Unit =
    spec.level.foreach { level =>
      val intLevel = level.toInt
      levelConfigs.get(spec.objectType).foreach { cfg =>
        val triangles = cfg.estimateTriangles(intLevel)
        if intLevel >= cfg.warnLevel then
          logger.warn(s"${spec.objectType} level $intLevel may be slow (~${triangles / 1000}K triangles)")
        if intLevel > cfg.maxLevel then
          logger.error(s"${spec.objectType} level $intLevel exceeds recommended maximum (${cfg.maxLevel})")
      }
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
    PlaneConfigurer.configurePlanes(renderer, environment.planes.toArray)
    sceneConfigurator.configureCamera(renderer)
    buildScene4DTrackedOrFallback(objectSpecs, renderer)
      .flatMap { _ =>
        Try {
          renderer.setRenderConfig(renderConfig)
          renderer.setCausticsConfig(config.caustics)
          configureOutputMode(renderer)
          environment.background.foreach(sceneConfigurator.setBackgroundColor(renderer, _))
          environment.fog.foreach(sceneConfigurator.setFog(renderer, _))
          environment.envMap.foreach { path =>
            val resolvedPath =
              if java.nio.file.Paths.get(path).isAbsolute then path
              else java.nio.file.Paths.get(config.execution.textureDir).resolve(path).toString
            try
              val idx = renderer.uploadTextureFromFile(resolvedPath)
              renderer.setEnvironmentMap(idx)
            catch
              case e: TextureUploadException =>
                logger.error(s"Failed to load environment map: $path: ${e.getMessage}")
          }
          environment.envMapVideo.foreach { envMapVideo =>
            TextureManager.loadInitialEnvMapVideo(
              envMapVideo,
              renderer,
              config.execution.textureDir
            ).foreach(renderer.setEnvironmentMap)
          }
          if environment.iblEnabled then
            renderer.setIBL(
              enabled  = true,
              strength = environment.iblStrength,
              samples  = environment.iblSamples
            )
          if crossVisible.get then addCrossGeometry(renderer)
          finalizeCreate()
        }
      }
      .recover { case e: Exception =>
        logger.error(s"Failed to create OptiX scene: ${e.getMessage}", e)
        GdxRuntime.exit()
      }.get

  private def addCrossGeometry(renderer: io.github.lene.optix.OptiXRenderer): Unit =
    val length    = config.cross.length
    val thickness = config.cross.thickness
    val baseMat   = config.cross.material.getOrElse(
      menger.common.Material(menger.common.Color(1f, 1f, 1f), roughness = 0.3f, metallic = 1.0f)
    )
    val axes = List(
      (menger.common.Vector[3](-length, 0f, 0f), menger.common.Vector[3](+length, 0f, 0f), 0),
      (menger.common.Vector[3](0f, -length, 0f), menger.common.Vector[3](0f, +length, 0f), 1),
      (menger.common.Vector[3](0f, 0f, -length), menger.common.Vector[3](0f, 0f, +length), 2)
    )
    axes.foreach { case (p0, p1, axisIdx) =>
      val mat = baseMat.copy(
        color = CrossAxisColors(axisIdx).color,
        ior   = 1.0f
      )
      InstanceId.fromNative(
        renderer.addCylinderInstance(p0, p1, thickness, mat),
        s"coordinate cross cylinder for axis $axisIdx"
      )
      val coneApex = menger.common.Vector[3](
        if axisIdx == 0 then length + CrossConeLength else p1(0),
        if axisIdx == 1 then length + CrossConeLength else p1(1),
        if axisIdx == 2 then length + CrossConeLength else p1(2)
      )
      InstanceId.fromNative(
        renderer.addConeInstance(coneApex, p1, CrossConeRadius, mat),
        s"coordinate cross cone for axis $axisIdx"
      )
    }

  private def toggleCross(): Unit =
    crossVisible.set(!crossVisible.get)
    rebuildScene()
    renderResources.markNeedsRender()
    GdxRuntime.requestRendering()

  private def resetTo4DDefaults(): Unit =
    logger.debug("Resetting 4D view to initial state")
    currentObjectSpecs.set(Some(objectSpecs))
    rebuildScene()
    renderResources.markNeedsRender()
    GdxRuntime.requestRendering()

  private def finalizeCreate(): Unit =
    val handler = OptiXKeyHandler(
      eventDispatcher,
      onReset = () => resetTo4DDefaults(),
      onToggleCross = () => toggleCross()
    )
    keyHandler.set(Some(handler))
    GdxRuntime.setInputProcessor(OptiXInputMultiplexer(cameraController, handler))
    GdxRuntime.setContinuousRendering(false)
    GdxRuntime.requestRendering()
    if execution.timeout > 0 then startExitTimer(execution.timeout)

  /** Build the initial scene with builder from [[GeometryRegistry.builderFor]] — the single
    * source of truth for type → builder dispatch. Captures per-spec instance/slot indices
    * for 4D-rotation fast paths where applicable. */
  private def buildScene4DTrackedOrFallback(
    specs: List[ObjectSpec],
    renderer: io.github.lene.optix.OptiXRenderer
  ): Try[Unit] =
    val builderOpt = GeometryRegistry.builderFor(specs, textureDir)
    builderOpt match
      case Some(builder) =>
        builder match
          case _: TriangleMeshSceneBuilder =>
            buildTriangleMesh4DTracked(specs, renderer)
          case _: Menger4DSceneBuilder =>
            build4DTracked(specs, renderer, (recorder: (Int, InstanceId) => Unit) =>
              new Menger4DSceneBuilder(textureDir, menger4DRecorder = recorder),
              (specs, ids) => Scene4DCache.Menger4D(Menger4DState(specs, ids)))
          case _: Sierpinski4DSceneBuilder =>
            build4DTracked(specs, renderer, (recorder: (Int, InstanceId) => Unit) =>
              new Sierpinski4DSceneBuilder(textureDir, sierpinski4DRecorder = recorder),
              (specs, ids) => Scene4DCache.Sierpinski4D(Sierpinski4DState(specs, ids)))
          case _: Hexadecachoron4DSceneBuilder =>
            build4DTracked(specs, renderer, (recorder: (Int, InstanceId) => Unit) =>
              new Hexadecachoron4DSceneBuilder(textureDir, hexadecachoron4DRecorder = recorder),
              (specs, ids) => Scene4DCache.Hexadecachoron4D(Hexadecachoron4DState(specs, ids)))
          case _ =>
            scene4DCache.set(Scene4DCache.Empty)
            builder.validateAndBuild(
              specs, renderer, computeEffectiveMaxInstances(builder, specs))
      case None =>
        scene4DCache.set(Scene4DCache.Empty)
        buildSceneFromSpecs(specs, renderer)

  private def buildTriangleMesh4DTracked(
    specs: List[ObjectSpec],
    renderer: io.github.lene.optix.OptiXRenderer
  ): Try[Unit] =
    val slotsBuf: scala.collection.mutable.Map[Int, ArrayBuffer[Int]] =
      scala.collection.mutable.Map.empty
    val builder = TriangleMeshSceneBuilder(
      textureDir,
      mesh4DRecorder = (specIdx: Int, slotIdx: Int) =>
        slotsBuf.getOrElseUpdate(specIdx, ArrayBuffer.empty[Int]) += slotIdx
    )(using profilingConfig)
    val maxInst = computeEffectiveMaxInstances(builder, specs)
    val result = builder.validateAndBuild(specs, renderer, maxInst)
    result.foreach { _ =>
      if slotsBuf.size == specs.size then
        val slotsPerSpec = specs.indices.map(i =>
          slotsBuf.getOrElse(i, ArrayBuffer.empty[Int]).toVector
        ).toVector
        scene4DCache.set(Scene4DCache.Gpu(WithAnimation.Anim4DState(specs, slotsPerSpec)))
      else scene4DCache.set(Scene4DCache.Empty)
    }
    result.recover { case _ => scene4DCache.set(Scene4DCache.Empty) }
    result

  private def build4DTracked(
    specs: List[ObjectSpec],
    renderer: io.github.lene.optix.OptiXRenderer,
    makeBuilder: ((Int, InstanceId) => Unit) => SceneBuilder,
    makeCache: (List[ObjectSpec], Vector[Vector[InstanceId]]) => Scene4DCache
  ): Try[Unit] =
    val instancesBuf: scala.collection.mutable.Map[Int, ArrayBuffer[InstanceId]] =
      scala.collection.mutable.Map.empty
    val recorder: (Int, InstanceId) => Unit =
      (specIdx, instanceId) =>
        instancesBuf.getOrElseUpdate(specIdx, ArrayBuffer.empty[InstanceId]) += instanceId
    val builder = makeBuilder(recorder)
    val maxInst = computeEffectiveMaxInstances(builder, specs)
    val result = builder.validateAndBuild(specs, renderer, maxInst)
    result.foreach { _ =>
      if instancesBuf.size == specs.size then
        val ids = specs.indices.map(i =>
          instancesBuf.getOrElse(i, ArrayBuffer.empty[InstanceId]).toVector
        ).toVector
        scene4DCache.set(makeCache(specs, ids))
      else scene4DCache.set(Scene4DCache.Empty)
    }
    result.recover { case _ => scene4DCache.set(Scene4DCache.Empty) }
    result

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
          renderer.clearAllInstances()
          buildScene4DTrackedOrFallback(specs, renderer).get
          if crossVisible.get then addCrossGeometry(renderer)
          cameraState.updateCamera(renderer, savedEye.toVector3, savedLookAt.toVector3, savedUp.toVector3)
          logger.debug("Scene rebuild complete")
        }.recover { case e =>
          scene4DCache.set(Scene4DCache.Empty)
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

  private val lastRenderResult = new AtomicReference[Option[RenderResult]](None)

  private def renderWithStats(width: Int, height: Int): Array[Byte] =
    rendererWrapper.renderSceneWithStats(ImageSize(width, height)) match
      case None =>
        logger.error("OptiX rendering failed - renderWithStats returned None")
        Array.emptyByteArray
      case Some(result) =>
        lastRenderResult.set(Some(result))
        val stats = result.stats
        logger.info(
          f"Frame: ${stats.frameMs}%.1f ms (${stats.msPerMray}%.2f ms/Mray) | " +
          s"primary=${stats.primaryRays} total=${stats.totalRays} " +
          s"reflected=${stats.reflectedRays} refracted=${stats.refractedRays} " +
          s"shadow=${stats.shadowRays} aa=${stats.aaRays} spectral=${stats.spectralRays} " +
          s"depth=${stats.minDepthReached}-${stats.maxDepthReached}"
        )
        result.image

  private def writeStatsJson(path: String): Unit =
    lastRenderResult.get() match
      case None =>
        logger.warn(s"No render result available; stats file not written: $path")
      case Some(result) =>
        val frameMs       = result.stats.frameMs
        val totalRays     = result.stats.totalRays
        val primaryRays   = result.stats.primaryRays
        val reflectedRays = result.stats.reflectedRays
        val refractedRays = result.stats.refractedRays
        val shadowRays    = result.stats.shadowRays
        val aaRays        = result.stats.aaRays
        val spectralRays  = result.stats.spectralRays
        val msPerMray     = result.stats.msPerMray
        val json =
          s"""|{
              |  "frameMs": $frameMs,
              |  "totalRays": $totalRays,
              |  "primaryRays": $primaryRays,
              |  "reflectedRays": $reflectedRays,
              |  "refractedRays": $refractedRays,
              |  "shadowRays": $shadowRays,
              |  "aaRays": $aaRays,
              |  "spectralRays": $spectralRays,
              |  "msPerMray": $msPerMray
              |}""".stripMargin
        Try {
          val p = Paths.get(path).toAbsolutePath
          Option(p.getParent).foreach(parent => Files.createDirectories(parent))
          Files.writeString(p, json)
          logger.info(s"Stats written to $p")
        }.failed.foreach { e =>
          logger.error(s"Failed to write stats to $path: ${e.getMessage}", e)
        }

  override def resize(width: Int, height: Int): Unit = {}

  override def dispose(): Unit =
    logger.debug("Disposing InteractiveEngine")
    execution.statsJsonPath.foreach(writeStatsJson)
    super.dispose()
