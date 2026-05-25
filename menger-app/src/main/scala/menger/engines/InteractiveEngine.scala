package menger.engines

import java.util.concurrent.atomic.AtomicReference

import scala.collection.mutable.ArrayBuffer
import scala.util.Try

import com.badlogic.gdx.graphics.GL20
import com.typesafe.scalalogging.LazyLogging
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
import menger.engines.scene.Hexadecachoron4DSceneBuilder
import menger.engines.scene.Menger4DSceneBuilder
import menger.engines.scene.MeshFactory
import menger.engines.scene.SceneBuilder
import menger.engines.scene.Sierpinski4DSceneBuilder
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

  override protected def renderConfig: menger.common.RenderConfig = config.render

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
  private case class Menger4DState(specs: List[ObjectSpec], instancesPerSpec: Vector[Vector[Int]])

  /** Per-spec instanceId mapping for the sierpinski4d rotation fast path. */
  private case class Sierpinski4DState(specs: List[ObjectSpec], instancesPerSpec: Vector[Vector[Int]])

  /** Per-spec instanceId mapping for the hexadecachoron4d rotation fast path. */
  private case class Hexadecachoron4DState(specs: List[ObjectSpec], instancesPerSpec: Vector[Vector[Int]])

  /** Cached slot/instance indices for 4D-rotation fast paths.
    * gpu/cpu: triangle mesh paths. menger4d/sierpinski4d/hexadecachoron4d: IFS instance paths. */
  private case class Scene4DCache(
    gpu:              Option[WithAnimation.Anim4DState] = None,
    cpu:              Option[WithAnimation.Anim4DState] = None,
    menger4d:         Option[Menger4DState]             = None,
    sierpinski4d:     Option[Sierpinski4DState]         = None,
    hexadecachoron4d: Option[Hexadecachoron4DState]     = None
  )
  private val scene4DCache: AtomicReference[Scene4DCache] =
    new AtomicReference(Scene4DCache())

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
        tryRotation4DCpuFastPath(specs, rendererWrapper.renderer) ||
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
    renderer: menger.optix.OptiXRenderer
  ): Boolean =
    if !renderConfig.gpuProject4D then false
    else scene4DCache.get.gpu match
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
          scene4DCache.set(scene4DCache.get.copy(gpu = Some(prev.copy(specs = newSpecs))))
          true

  /** Per-mesh CPU re-projection update path. Mirrors `tryRotation4DFastPath` but
    * for CPU-projected meshes. Re-projects each spec on CPU via `MeshFactory.create`
    * and updates the existing mesh slot via `renderer.updateCpuTriangleMesh` —
    * avoiding the full `clearAllInstances()` + rebuild that causes a hang when
    * `gpuProject4D` is off (H-mixed-frac-int-interactive-hang).
    * Returns true iff the cpu4DState slot map is populated and every spec is
    * a 4D-projected type (non-fractional, since fractional merging complicates
    * slot mapping for now). */
  private def tryRotation4DCpuFastPath(
    newSpecs: List[ObjectSpec],
    renderer: menger.optix.OptiXRenderer
  ): Boolean =
    if renderConfig.gpuProject4D then false  // GPU path handles this
    else scene4DCache.get.cpu match
      case None => false
      case Some(prev) =>
        if !WithAnimation.specsDifferOnlyIn4DProjection(prev.specs, newSpecs) then false
        else
          val success = prev.specs.lazyZip(newSpecs).lazyZip(prev.slotsPerSpec).forall {
            case (prevSpec, newSpec, slots) =>
              if prevSpec.projection4D == newSpec.projection4D then true
              else
                val meshData = MeshFactory.create(newSpec)
                slots.forall { slot =>
                  Try(renderer.updateCpuTriangleMesh(slot, meshData)).isSuccess
                }
          }
          if success then
            scene4DCache.set(scene4DCache.get.copy(cpu = Some(prev.copy(specs = newSpecs))))
          success

  /** Menger4D fast path: update projection params on each recorded instance directly.
    * Returns true iff the menger4d slot map is populated and the change is purely
    * a Projection4DSpec delta — skipping the full geometry rebuild. */
  private def tryMenger4DFastPath(
    newSpecs: List[ObjectSpec],
    renderer: menger.optix.OptiXRenderer
  ): Boolean =
    scene4DCache.get.menger4d match
      case None => false
      case Some(prev) =>
        if !WithAnimation.specsDifferOnlyIn4DProjection(prev.specs, newSpecs) then false
        else
          prev.specs.lazyZip(newSpecs).lazyZip(prev.instancesPerSpec).foreach {
            case (prevSpec, newSpec, instanceIds) =>
              if prevSpec.projection4D != newSpec.projection4D then
                val proj = newSpec.projection4D.getOrElse(Projection4DSpec.default)
                instanceIds.foreach { instanceId =>
                  renderer.updateMenger4DProjection(
                    instanceId,
                    eyeW = proj.eyeW, screenW = proj.screenW,
                    rotXW = proj.rotXW, rotYW = proj.rotYW, rotZW = proj.rotZW
                  )
                }
          }
          scene4DCache.set(scene4DCache.get.copy(menger4d = Some(prev.copy(specs = newSpecs))))
          true

  private def trySierpinski4DFastPath(
    newSpecs: List[ObjectSpec],
    renderer: menger.optix.OptiXRenderer
  ): Boolean =
    scene4DCache.get.sierpinski4d match
      case None => false
      case Some(prev) =>
        if !WithAnimation.specsDifferOnlyIn4DProjection(prev.specs, newSpecs) then false
        else
          prev.specs.lazyZip(newSpecs).lazyZip(prev.instancesPerSpec).foreach {
            case (prevSpec, newSpec, instanceIds) =>
              if prevSpec.projection4D != newSpec.projection4D then
                val proj = newSpec.projection4D.getOrElse(Projection4DSpec.default)
                instanceIds.foreach { instanceId =>
                  renderer.updateSierpinski4DProjection(
                    instanceId,
                    eyeW = proj.eyeW, screenW = proj.screenW,
                    rotXW = proj.rotXW, rotYW = proj.rotYW, rotZW = proj.rotZW
                  )
                }
          }
          scene4DCache.set(scene4DCache.get.copy(sierpinski4d = Some(prev.copy(specs = newSpecs))))
          true

  private def tryHexadecachoron4DFastPath(
    newSpecs: List[ObjectSpec],
    renderer: menger.optix.OptiXRenderer
  ): Boolean =
    scene4DCache.get.hexadecachoron4d match
      case None => false
      case Some(prev) =>
        if !WithAnimation.specsDifferOnlyIn4DProjection(prev.specs, newSpecs) then false
        else
          prev.specs.lazyZip(newSpecs).lazyZip(prev.instancesPerSpec).foreach {
            case (prevSpec, newSpec, instanceIds) =>
              if prevSpec.projection4D != newSpec.projection4D then
                val proj = newSpec.projection4D.getOrElse(Projection4DSpec.default)
                instanceIds.foreach { instanceId =>
                  renderer.updateHexadecachoron4DProjection(
                    instanceId,
                    eyeW = proj.eyeW, screenW = proj.screenW,
                    rotXW = proj.rotXW, rotYW = proj.rotYW, rotZW = proj.rotZW
                  )
                }
          }
          scene4DCache.set(scene4DCache.get.copy(hexadecachoron4d = Some(prev.copy(specs = newSpecs))))
          true

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
          environment.background.foreach(sceneConfigurator.setBackgroundColor(renderer, _))
          environment.fog.foreach(sceneConfigurator.setFog(renderer, _))
          environment.envMap.foreach { path =>
            val resolvedPath =
              if java.nio.file.Paths.get(path).isAbsolute then path
              else java.nio.file.Paths.get(config.execution.textureDir).resolve(path).toString
            val idx = renderer.uploadTextureFromFile(resolvedPath)
            if idx >= 0 then renderer.setEnvironmentMap(idx)
            else logger.error(s"Failed to load environment map: $path")
          }
          if crossVisible.get then addCrossGeometry(renderer)
          finalizeCreate()
        }
      }
      .recover { case e: Exception =>
        logger.error(s"Failed to create OptiX scene: ${e.getMessage}", e)
        GdxRuntime.exit()
      }.get

  private def addCrossGeometry(renderer: menger.optix.OptiXRenderer): Unit =
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
      renderer.addCylinderInstance(p0, p1, thickness, mat)
      val coneApex = menger.common.Vector[3](
        if axisIdx == 0 then length + CrossConeLength else p1(0),
        if axisIdx == 1 then length + CrossConeLength else p1(1),
        if axisIdx == 2 then length + CrossConeLength else p1(2)
      )
      renderer.addConeInstance(coneApex, p1, CrossConeRadius, mat)
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

  /** Build the initial scene; if it is 4D-only triangle meshes and `gpuProject4D`
    * is on, capture per-spec slot indices so subsequent rotation events can hit
    * the fast path. When `gpuProject4D` is off and the scene is 4D-only and
    * non-fractional, record CPU mesh slots for the analogous CPU fast path.
    * Otherwise fall back to the generic build and clear cached fast-path state.
    * Mirrors `WithAnimation.buildAnim4DTrackedOrFallback`. */
  private def buildScene4DTrackedOrFallback(
    specs: List[ObjectSpec],
    renderer: menger.optix.OptiXRenderer
  ): Try[Unit] =
    if renderConfig.gpuProject4D && WithAnimation.is4DOnlyTriangleMeshScene(specs)
        && !specs.exists(_.hasEdgeRendering) then
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
          scene4DCache.set(Scene4DCache(gpu = Some(WithAnimation.Anim4DState(specs, slotsPerSpec))))
        else
          scene4DCache.set(Scene4DCache())
      }
      result.recover { case _ => scene4DCache.set(Scene4DCache()) }
      result
    else if !renderConfig.gpuProject4D && isCpu4DFastPathEligible(specs) then
      val builder = TriangleMeshSceneBuilder(textureDir, gpuProject4D = false)(using profilingConfig)
      val effectiveMaxInstances = computeEffectiveMaxInstances(builder, specs)
      val result = builder.buildScene(specs, renderer, effectiveMaxInstances)
      result.foreach { _ =>
        // CPU path: one mesh per spec (fractional = merged single mesh),
        // slots are assigned sequentially starting from 0 after clearAllInstances.
        val slotsPerSpec = specs.indices.map(i => Vector(i)).toVector
        scene4DCache.set(Scene4DCache(cpu = Some(WithAnimation.Anim4DState(specs, slotsPerSpec))))
      }
      result.recover { case _ => scene4DCache.set(Scene4DCache()) }
      result
    else if specs.forall(s => ObjectType.isMenger4D(s.objectType)) then
      val instancesBuf: scala.collection.mutable.Map[Int, ArrayBuffer[Int]] =
        scala.collection.mutable.Map.empty
      val builder = Menger4DSceneBuilder(
        textureDir,
        menger4DRecorder = (specIdx: Int, instanceId: Int) =>
          instancesBuf.getOrElseUpdate(specIdx, ArrayBuffer.empty[Int]) += instanceId
      )
      val effectiveMaxInstances = computeEffectiveMaxInstances(builder, specs)
      val result = builder.buildScene(specs, renderer, effectiveMaxInstances)
      result.foreach { _ =>
        if instancesBuf.size == specs.size then
          val instancesPerSpec = specs.indices.map(i =>
            instancesBuf.getOrElse(i, ArrayBuffer.empty[Int]).toVector
          ).toVector
          scene4DCache.set(Scene4DCache(menger4d = Some(Menger4DState(specs, instancesPerSpec))))
        else
          scene4DCache.set(Scene4DCache())
      }
      result.recover { case _ => scene4DCache.set(Scene4DCache()) }
      result
    else if specs.forall(s => ObjectType.isSierpinski4D(s.objectType)) then
      val instancesBuf: scala.collection.mutable.Map[Int, ArrayBuffer[Int]] =
        scala.collection.mutable.Map.empty
      val builder = Sierpinski4DSceneBuilder(
        textureDir,
        sierpinski4DRecorder = (specIdx: Int, instanceId: Int) =>
          instancesBuf.getOrElseUpdate(specIdx, ArrayBuffer.empty[Int]) += instanceId
      )
      val effectiveMaxInstances = computeEffectiveMaxInstances(builder, specs)
      val result = builder.buildScene(specs, renderer, effectiveMaxInstances)
      result.foreach { _ =>
        if instancesBuf.size == specs.size then
          val instancesPerSpec = specs.indices.map(i =>
            instancesBuf.getOrElse(i, ArrayBuffer.empty[Int]).toVector
          ).toVector
          scene4DCache.set(Scene4DCache(sierpinski4d = Some(Sierpinski4DState(specs, instancesPerSpec))))
        else
          scene4DCache.set(Scene4DCache())
      }
      result.recover { case _ => scene4DCache.set(Scene4DCache()) }
      result
    else if specs.forall(s => ObjectType.isHexadecachoron4D(s.objectType)) then
      val instancesBuf: scala.collection.mutable.Map[Int, ArrayBuffer[Int]] =
        scala.collection.mutable.Map.empty
      val builder = Hexadecachoron4DSceneBuilder(
        textureDir,
        hexadecachoron4DRecorder = (specIdx: Int, instanceId: Int) =>
          instancesBuf.getOrElseUpdate(specIdx, ArrayBuffer.empty[Int]) += instanceId
      )
      val effectiveMaxInstances = computeEffectiveMaxInstances(builder, specs)
      val result = builder.buildScene(specs, renderer, effectiveMaxInstances)
      result.foreach { _ =>
        if instancesBuf.size == specs.size then
          val instancesPerSpec = specs.indices.map(i =>
            instancesBuf.getOrElse(i, ArrayBuffer.empty[Int]).toVector
          ).toVector
          scene4DCache.set(Scene4DCache(hexadecachoron4d = Some(Hexadecachoron4DState(specs, instancesPerSpec))))
        else
          scene4DCache.set(Scene4DCache())
      }
      result.recover { case _ => scene4DCache.set(Scene4DCache()) }
      result
    else
      scene4DCache.set(Scene4DCache())
      buildSceneFromSpecs(specs, renderer)

  /** True iff this scene is eligible for the CPU 4D fast path:
    * all specs are 4D-projected triangle-mesh types. Fractional specs are
    * allowed because the CPU path merges them into a single mesh (1 slot/spec).
    * Mixed scenes (4D + non-4D) are excluded.
    * Scenes with edge rendering are excluded — those must use TesseractEdgeSceneBuilder. */
  private def isCpu4DFastPathEligible(specs: List[ObjectSpec]): Boolean =
    WithAnimation.is4DOnlyTriangleMeshScene(specs) && !specs.exists(_.hasEdgeRendering)

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
          if crossVisible.get then addCrossGeometry(renderer)
          cameraState.updateCamera(renderer, savedEye.toVector3, savedLookAt.toVector3, savedUp.toVector3)
          // After a full rebuild, instance/slot IDs restart from 0 sequentially.
          // Re-populate fast path caches so subsequent rotations stay on the fast path.
          if !renderConfig.gpuProject4D && isCpu4DFastPathEligible(specs) then
            val slotsPerSpec = specs.indices.map(i => Vector(i)).toVector
            scene4DCache.set(scene4DCache.get.copy(
              cpu              = Some(WithAnimation.Anim4DState(specs, slotsPerSpec)),
              menger4d         = None,
              sierpinski4d     = None,
              hexadecachoron4d = None
            ))
          else if specs.forall(s => ObjectType.isMenger4D(s.objectType)) then
            val instancesPerSpec = specs.indices.map(i => Vector(i)).toVector
            scene4DCache.set(scene4DCache.get.copy(
              cpu              = None,
              menger4d         = Some(Menger4DState(specs, instancesPerSpec)),
              sierpinski4d     = None,
              hexadecachoron4d = None
            ))
          else if specs.forall(s => ObjectType.isSierpinski4D(s.objectType)) then
            val instancesPerSpec = specs.indices.map(i => Vector(i)).toVector
            scene4DCache.set(scene4DCache.get.copy(
              cpu              = None,
              menger4d         = None,
              sierpinski4d     = Some(Sierpinski4DState(specs, instancesPerSpec)),
              hexadecachoron4d = None
            ))
          else if specs.forall(s => ObjectType.isHexadecachoron4D(s.objectType)) then
            val instancesPerSpec = specs.indices.map(i => Vector(i)).toVector
            scene4DCache.set(scene4DCache.get.copy(
              cpu              = None,
              menger4d         = None,
              sierpinski4d     = None,
              hexadecachoron4d = Some(Hexadecachoron4DState(specs, instancesPerSpec))
            ))
          else
            scene4DCache.set(scene4DCache.get.copy(cpu = None, menger4d = None, sierpinski4d = None, hexadecachoron4d = None))
          logger.debug("Scene rebuild complete")
        }.recover { case e =>
          scene4DCache.set(scene4DCache.get.copy(cpu = None, menger4d = None, sierpinski4d = None, hexadecachoron4d = None))
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
    rendererWrapper.renderSceneWithStats(ImageSize(width, height)) match
      case None =>
        logger.error("OptiX rendering failed - renderWithStats returned None")
        Array.emptyByteArray
      case Some(result) =>
        val stats = result.stats
        logger.info(
          f"Frame: ${stats.frameMs}%.1f ms (${stats.msPerMray}%.2f ms/Mray) | " +
          s"primary=${stats.primaryRays} total=${stats.totalRays} " +
          s"reflected=${stats.reflectedRays} refracted=${stats.refractedRays} " +
          s"shadow=${stats.shadowRays} aa=${stats.aaRays} " +
          s"depth=${stats.minDepthReached}-${stats.maxDepthReached}"
        )
        result.image

  override def resize(width: Int, height: Int): Unit = {}

  override def dispose(): Unit =
    logger.debug("Disposing InteractiveEngine")
    super.dispose()
