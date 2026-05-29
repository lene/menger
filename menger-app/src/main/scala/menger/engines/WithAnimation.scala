package menger.engines

import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

import scala.collection.mutable.ArrayBuffer
import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.graphics.GL20
import com.typesafe.scalalogging.LazyLogging
import io.github.lene.optix.OptiXRenderer
import menger.ObjectSpec
import menger.Projection4DSpec
import menger.Vector3Extensions.toVector3
import menger.common.CausticsConfig
import menger.common.ImageSize
import menger.common.ObjectType
import menger.common.RenderConfig
import menger.config.TAnimationConfig
import menger.dsl.Scene
import menger.engines.scene.TriangleMeshSceneBuilder
import menger.input.GdxRuntime

trait WithAnimation extends RenderEngine with SavesScreenshots with LazyLogging:
  self: BaseEngine =>

  protected def sceneFunction: Float => Scene
  protected def animConfig: TAnimationConfig
  protected def renderConfig: RenderConfig
  protected def causticsConfig: CausticsConfig
  protected def firstFrameConfigs: SceneConverter.SceneConfigs

  protected val frameCounter: AtomicInteger = new AtomicInteger(0)

  /** Cached state for the GPU 4D-projection animation fast path: when the only
    * frame-to-frame change is `Projection4DSpec` on 4D-projected specs, we
    * call `renderer.updateMesh4DProjection` instead of clearAllInstances+rebuild.
    * Set only when `renderConfig.gpuProject4D` is on AND the scene is purely
    * 4D-projected triangle meshes. */
  private val anim4DState: AtomicReference[Option[WithAnimation.Anim4DState]] =
    new AtomicReference(None)

  abstract override def create(): Unit =
    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureLights(renderer)
    sceneConfigurator.configureCamera(renderer)
    buildSceneFromConfigs(firstFrameConfigs, renderer).recover { case e: Exception =>
      logger.error(s"Failed to create initial scene: ${e.getMessage}", e)
      GdxRuntime.exit()
    }.get
    renderer.setRenderConfig(renderConfig)
    renderer.setCausticsConfig(firstFrameConfigs.caustics)
    PlaneConfigurer.configurePlanes(renderer, firstFrameConfigs.planes.toArray)
    if firstFrameConfigs.toneMappingOperator != 0 then
      renderer.setToneMapping(firstFrameConfigs.toneMappingOperator, firstFrameConfigs.toneMappingExposure)
    firstFrameConfigs.envMap.foreach { path =>
      val resolvedPath =
        if java.nio.file.Paths.get(path).isAbsolute then path
        else java.nio.file.Paths.get(textureDir).resolve(path).toString
      val idx = renderer.uploadTextureFromFile(resolvedPath)
      if idx >= 0 then renderer.setEnvironmentMap(idx)
      else logger.error(s"Failed to load environment map: $path")
    }
    if firstFrameConfigs.iblEnabled then
      renderer.setIBL(
        enabled  = true,
        strength = firstFrameConfigs.iblStrength,
        samples  = firstFrameConfigs.iblSamples,
      )
    if firstFrameConfigs.accumulationFrames > 1 then
      renderer.setAccumulationFrames(firstFrameConfigs.accumulationFrames)
    GdxRuntime.setContinuousRendering(true)

  abstract override def render(): Unit =
    GdxRuntime.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)
    val width  = GdxRuntime.width
    val height = GdxRuntime.height
    val frame  = frameCounter.get()

    if width > 0 && height > 0 && frame < animConfig.frames then
      val t = animConfig.tForFrame(frame)
      logger.info(s"Rendering frame ${frame + 1}/${animConfig.frames} (t=$t)")
      Try(sceneFunction(t)) match
        case Failure(e) =>
          logger.error(
            s"Scene function threw for frame ${frame + 1}/${animConfig.frames}" +
            s" (t=$t): ${e.getMessage}",
            e
          )
          frameCounter.incrementAndGet()
        case scala.util.Success(dslScene) =>
          val configs  = SceneConverter.convert(dslScene, causticsConfig)
          val renderer = rendererWrapper.renderer
          val newSpecs = configs.scene.objectSpecs.getOrElse(List.empty)
          val fastPathTaken = tryAnim4DFastPath(newSpecs, renderer)
          if !fastPathTaken then
            renderer.clearAllInstances()
            buildAnim4DTrackedOrFallback(configs, newSpecs, renderer).recover { case e: Exception =>
              logger.error(s"Failed to build scene for frame $frame (t=$t): ${e.getMessage}", e)
            }
          PlaneConfigurer.configurePlanes(renderer, configs.planes.toArray)
          configs.background.foreach(c => sceneConfigurator.setBackgroundColor(renderer, c))
          configs.fog.foreach(f => sceneConfigurator.setFog(renderer, f))
          cameraState.updateCamera(
            renderer,
            configs.camera.position.toVector3,
            configs.camera.lookAt.toVector3,
            configs.camera.up.toVector3
          )
          cameraState.updateCameraAspectRatio(renderer, ImageSize(width, height))
          val rgbaBytes = rendererWrapper.renderScene(ImageSize(width, height))
          renderResources.renderToScreen(rgbaBytes, width, height)
          saveImage()
          frameCounter.incrementAndGet()
          ()
    else if frame >= animConfig.frames then
      logger.info(s"Animation complete: ${animConfig.frames} frames rendered")
      onAnimationComplete()
      GdxRuntime.exit()

  /** If conditions are met, drive frame N from frame N−1 via per-mesh
    * `updateMesh4DProjection` calls and return true. Otherwise return false
    * and let the caller take the rebuild path. */
  private def tryAnim4DFastPath(newSpecs: List[ObjectSpec], renderer: OptiXRenderer): Boolean =
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

  /** Rebuild path: when the scene is 4D-only triangle-mesh AND `gpuProject4D`
    * is on, build via a recorder-equipped TriangleMeshSceneBuilder so
    * subsequent frames can attempt the fast path. Otherwise fall back to the
    * generic `buildSceneFromConfigs` path and clear any cached fast-path
    * state. */
  private def buildAnim4DTrackedOrFallback(
    configs: SceneConverter.SceneConfigs,
    newSpecs: List[ObjectSpec],
    renderer: OptiXRenderer
  ): Try[Unit] =
    if renderConfig.gpuProject4D && WithAnimation.is4DOnlyTriangleMeshScene(newSpecs) then
      val slotsBuf: scala.collection.mutable.Map[Int, ArrayBuffer[Int]] =
        scala.collection.mutable.Map.empty
      val builder = TriangleMeshSceneBuilder(
        textureDir, gpuProject4D = true,
        mesh4DRecorder = (specIdx: Int, slotIdx: Int) =>
          slotsBuf.getOrElseUpdate(specIdx, ArrayBuffer.empty[Int]) += slotIdx
      )(using profilingConfig)
      val result = builder.buildScene(newSpecs, renderer, newSpecs.length)
      result.foreach { _ =>
        if slotsBuf.size == newSpecs.size then
          val slotsPerSpec = newSpecs.indices.map(i =>
            slotsBuf.getOrElse(i, ArrayBuffer.empty[Int]).toVector
          ).toVector
          anim4DState.set(Some(WithAnimation.Anim4DState(newSpecs, slotsPerSpec)))
        else
          anim4DState.set(None)
      }
      val recovered = result.recover { case _ => anim4DState.set(None) }
      recovered
    else
      anim4DState.set(None)
      buildSceneFromConfigs(configs, renderer)

object WithAnimation:
  /** Per-mesh-slot bookkeeping for the GPU 4D projection animation fast path.
    * `slotsPerSpec(i)` lists the renderer slot indices that belong to spec i —
    * usually a single slot, but two for fractional 4D sponges (level n + level n+1). */
  final case class Anim4DState(specs: List[ObjectSpec], slotsPerSpec: Vector[Vector[Int]])

  /** Eligibility: every spec is a 4D-projected type (tesseract, tesseract-sponge,
    * tesseract-sponge-2). Recursive-IAS sponges and non-4D meshes block the
    * fast path because their slot mapping isn't 1:1. */
  def is4DOnlyTriangleMeshScene(specs: List[ObjectSpec]): Boolean =
    specs.nonEmpty && specs.forall(s => ObjectType.isProjected4D(s.objectType))

  /** True iff the two spec lists have the same length and differ only in
    * `Projection4DSpec` (and at least one element actually differs there). */
  def specsDifferOnlyIn4DProjection(prev: List[ObjectSpec], next: List[ObjectSpec]): Boolean =
    if prev.length != next.length then false
    else
      val pairs = prev.zip(next)
      val anyDiffs = pairs.exists { case (a, b) => a.projection4D != b.projection4D }
      val onlyProjDiffs = pairs.forall { case (a, b) =>
        a.copy(projection4D = None) == b.copy(projection4D = None)
      }
      anyDiffs && onlyProjDiffs
