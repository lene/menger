package menger.engines

import java.nio.file.Path
import java.nio.file.Paths
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

import scala.collection.mutable.ArrayBuffer
import scala.util.Failure
import scala.util.Success
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
import menger.engines.scene.TextureManager
import menger.engines.scene.TriangleMeshSceneBuilder
import menger.geometry.VideoLoader
import menger.input.GdxRuntime
import menger.video.AnimationTimeRange
import menger.video.EnvMapVideo
import menger.video.VideoPlaybackTime
import menger.video.VideoTexture

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
    * Set only when the scene is purely 4D-projected triangle meshes. */
  private val anim4DState: AtomicReference[Option[WithAnimation.Anim4DState]] =
    new AtomicReference(None)
  private val videoTextureState: AtomicReference[Option[WithAnimation.VideoTextureState]] =
    new AtomicReference(None)
  private val envMapVideoState: AtomicReference[Option[WithAnimation.EnvMapVideoState]] =
    new AtomicReference(None)

  abstract override def create(): Unit =
    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureLights(renderer)
    sceneConfigurator.configureCamera(renderer)
    val firstSpecs = firstFrameConfigs.scene.objectSpecs.getOrElse(List.empty)
    buildSceneTrackingVideoTextures(firstFrameConfigs, firstSpecs, renderer).recover {
      case e: Exception =>
        logger.error(s"Failed to create initial scene: ${e.getMessage}", e)
        GdxRuntime.exit()
    }.get
    renderer.setRenderConfig(renderConfig)
    renderer.setCausticsConfig(firstFrameConfigs.caustics)
    PlaneConfigurer.configurePlanes(renderer, firstFrameConfigs.planes.toArray)
    if firstFrameConfigs.toneMappingOperator != 0 then
      renderer.setToneMapping(firstFrameConfigs.toneMappingOperator, firstFrameConfigs.toneMappingExposure)
    configureStaticEnvironmentMap(firstFrameConfigs, renderer)
    configureEnvMapVideoForFrame(
      firstFrameConfigs,
      firstSpecs,
      renderer,
      animConfig.tForFrame(0)
    ).recover { case e: Exception =>
      logger.error(s"Failed to load environment-map video: ${e.getMessage}", e)
    }
    configureIBL(renderer, firstFrameConfigs)
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
          val videoSourcesPresent = WithAnimation.hasVideoSources(configs, newSpecs)
          val videoFastPathTaken = tryVideoFastPath(configs, newSpecs, renderer, t)
          val geometryFastPathTaken =
            if videoSourcesPresent && !videoFastPathTaken then false
            else tryAnim4DFastPath(newSpecs, renderer)
          val fastPathTaken = videoFastPathTaken || geometryFastPathTaken
          if !fastPathTaken then
            renderer.clearAllInstances()
            buildAnim4DTrackedOrFallback(configs, newSpecs, renderer).recover { case e: Exception =>
              logger.error(s"Failed to build scene for frame $frame (t=$t): ${e.getMessage}", e)
            }
            configureEnvMapVideoForFrame(configs, newSpecs, renderer, t).recover {
              case e: Exception =>
                logger.error(s"Failed to update environment-map video for t=$t: ${e.getMessage}", e)
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

  private def tryVideoFastPath(
    configs: SceneConverter.SceneConfigs,
    newSpecs: List[ObjectSpec],
    renderer: OptiXRenderer,
    t: Float
  ): Boolean =
    val needsVideoTextures = newSpecs.exists(_.videoTexture.nonEmpty)
    val needsEnvMapVideo = configs.envMapVideo.nonEmpty
    if !needsVideoTextures && !needsEnvMapVideo then false
    else
      val reusableVideoTextureState = videoTextureState.get.filter { state =>
        WithAnimation.specsCanReuseVideoTextureSlots(state.specs, newSpecs)
      }
      val reusableEnvMapVideoState = configs.envMapVideo.flatMap { envMapVideo =>
        envMapVideoState.get.filter { state =>
          WithAnimation.configsCanReuseEnvMapVideoSlot(
            state.specs,
            newSpecs,
            state.envMapVideo,
            envMapVideo
          )
        }
      }
      val videoTexturesReady = !needsVideoTextures || reusableVideoTextureState.nonEmpty
      val envMapVideoReady = !needsEnvMapVideo || reusableEnvMapVideoState.nonEmpty
      if !videoTexturesReady || !envMapVideoReady then false
      else
        val textureUpdate = reusableVideoTextureState
          .filter(_ => needsVideoTextures)
          .map(state => updateVideoTextureSlots(state.slots, renderer, t))
          .getOrElse(Success(()))
        val envMapUpdate = reusableEnvMapVideoState
          .filter(_ => needsEnvMapVideo)
          .map(state => updateEnvMapVideoSlot(state, configs, renderer, t))
          .getOrElse(Success(()))
        (textureUpdate, envMapUpdate) match
          case (Success(_), Success(_)) =>
            reusableVideoTextureState
              .filter(_ => needsVideoTextures)
              .foreach(state => videoTextureState.set(Some(state.copy(specs = newSpecs))))
            reusableEnvMapVideoState
              .filter(_ => needsEnvMapVideo)
              .foreach(state => envMapVideoState.set(Some(state.copy(specs = newSpecs))))
            true
          case (Failure(e), _) =>
            logger.error(s"Failed to update video texture frame for t=$t: ${e.getMessage}", e)
            false
          case (_, Failure(e)) =>
            logger.error(s"Failed to update environment-map video for t=$t: ${e.getMessage}", e)
            false

  private def updateVideoTextureSlots(
    slots: Vector[WithAnimation.VideoTextureSlot],
    renderer: OptiXRenderer,
    t: Float
  ): Try[Unit] = Try:
    val animationRange = AnimationTimeRange(animConfig.startT, animConfig.endT)
    slots.foreach { slot =>
      val resolvedPath = resolveVideoPath(slot.videoTexture)
      val loader = new VideoLoader(resolvedPath.toString)
      try
        val frameTime = VideoPlaybackTime.sampleTime(
          slot.videoTexture.playback,
          t,
          animationRange,
          loader.durationSeconds
        )
        val frameData = loader.frameAt(frameTime)
        slot.textureIndices.foreach { textureIndex =>
          renderer.updateTexture(textureIndex, frameData, loader.width, loader.height)
        }
      finally loader.close()
    }

  private def resolveVideoPath(videoTexture: VideoTexture): Path =
    resolveVideoPath(videoTexture.path)

  private def resolveVideoPath(path: String): Path =
    val videoPath = Paths.get(path)
    if videoPath.isAbsolute then videoPath else Paths.get(textureDir).resolve(videoPath)

  private def resolveEnvMapVideoPath(envMapVideo: EnvMapVideo): Path =
    resolveVideoPath(envMapVideo.path)

  private def configureStaticEnvironmentMap(
    configs: SceneConverter.SceneConfigs,
    renderer: OptiXRenderer
  ): Unit =
    configs.envMap.foreach { path =>
      val resolvedPath =
        if Paths.get(path).isAbsolute then path
        else Paths.get(textureDir).resolve(path).toString
      val idx = renderer.uploadTextureFromFile(resolvedPath)
      if idx >= 0 then renderer.setEnvironmentMap(idx)
      else logger.error(s"Failed to load environment map: $path")
    }

  private def configureIBL(
    renderer: OptiXRenderer,
    configs: SceneConverter.SceneConfigs
  ): Unit =
    if configs.iblEnabled || configs.envMapVideo.nonEmpty then
      renderer.setIBL(
        enabled  = configs.iblEnabled,
        strength = configs.iblStrength,
        samples  = configs.iblSamples,
      )

  private def configureEnvMapVideoForFrame(
    configs: SceneConverter.SceneConfigs,
    newSpecs: List[ObjectSpec],
    renderer: OptiXRenderer,
    t: Float
  ): Try[Unit] =
    configs.envMapVideo match
      case None =>
        envMapVideoState.set(None)
        Success(())
      case Some(envMapVideo) =>
        val existingState = envMapVideoState.get.filter { state =>
          state.envMapVideo.textureKey == envMapVideo.textureKey
        }
        val state = existingState
          .map(existing => Success(existing.copy(specs = newSpecs, envMapVideo = envMapVideo)))
          .getOrElse(uploadEnvMapVideoSlot(envMapVideo, renderer).map { textureIndex =>
            WithAnimation.EnvMapVideoState(newSpecs, envMapVideo, textureIndex)
          })
        state.flatMap { activeState =>
          updateEnvMapVideoSlot(activeState, configs, renderer, t).map { _ =>
            envMapVideoState.set(Some(activeState))
          }
        }

  private def uploadEnvMapVideoSlot(
    envMapVideo: EnvMapVideo,
    renderer: OptiXRenderer
  ): Try[Int] =
    TextureManager.loadInitialEnvMapVideo(envMapVideo, renderer, textureDir) match
      case Some(textureIndex) =>
        renderer.setEnvironmentMap(textureIndex)
        Success(textureIndex)
      case None =>
        Failure(IllegalStateException(
          s"Failed to upload environment-map video '${envMapVideo.path}'"
        ))

  private def updateEnvMapVideoSlot(
    state: WithAnimation.EnvMapVideoState,
    configs: SceneConverter.SceneConfigs,
    renderer: OptiXRenderer,
    t: Float
  ): Try[Unit] = Try:
    val resolvedPath = resolveEnvMapVideoPath(state.envMapVideo)
    val loader = new VideoLoader(resolvedPath.toString)
    try
      TextureManager.validateEquirectangularDimensions(
        loader.width,
        loader.height,
        state.envMapVideo.path
      )
      val animationRange = AnimationTimeRange(animConfig.startT, animConfig.endT)
      val frameTime = VideoPlaybackTime.sampleTime(
        state.envMapVideo.playback,
        t,
        animationRange,
        loader.durationSeconds
      )
      val frameData = loader.frameAt(frameTime)
      renderer.updateTexture(state.textureIndex, frameData, loader.width, loader.height)
      renderer.setEnvironmentMap(state.textureIndex)
      configureIBL(renderer, configs)
    finally loader.close()

  private def buildSceneTrackingVideoTextures(
    configs: SceneConverter.SceneConfigs,
    newSpecs: List[ObjectSpec],
    renderer: OptiXRenderer
  ): Try[Unit] =
    buildWithVideoTextureTracking(newSpecs):
      buildSceneFromConfigs(configs, renderer)

  private def buildWithVideoTextureTracking(
    newSpecs: List[ObjectSpec]
  )(body: => Try[Unit]): Try[Unit] =
    val observedSlots =
      scala.collection.mutable.LinkedHashMap.empty[String, (VideoTexture, ArrayBuffer[Int])]
    val result = TextureManager.withVideoTextureSlotObserver { (videoTexture, textureIndex) =>
      observedSlots.get(videoTexture.textureKey) match
        case Some((_, textureIndices)) =>
          textureIndices += textureIndex
        case None =>
          observedSlots.update(videoTexture.textureKey, videoTexture -> ArrayBuffer(textureIndex))
    } {
      body
    }
    result match
      case Success(_) =>
        val slots = observedSlots.valuesIterator.map { case (videoTexture, textureIndices) =>
          WithAnimation.VideoTextureSlot(videoTexture, textureIndices.toVector)
        }.toVector
        videoTextureState.set(
          if slots.nonEmpty then Some(WithAnimation.VideoTextureState(newSpecs, slots)) else None
        )
      case Failure(_) =>
        videoTextureState.set(None)
    result

  /** If conditions are met, drive frame N from frame N−1 via per-mesh
    * `updateMesh4DProjection` calls and return true. Otherwise return false
    * and let the caller take the rebuild path. */
  private def tryAnim4DFastPath(newSpecs: List[ObjectSpec], renderer: OptiXRenderer): Boolean =
    anim4DState.get match
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

  /** Rebuild path: when the scene is 4D-only triangle-mesh, build via a
    * recorder-equipped TriangleMeshSceneBuilder so subsequent frames can attempt
    * the fast path. Otherwise fall back to the generic `buildSceneFromConfigs`
    * path and clear any cached fast-path state. */
  private def buildAnim4DTrackedOrFallback(
    configs: SceneConverter.SceneConfigs,
    newSpecs: List[ObjectSpec],
    renderer: OptiXRenderer
  ): Try[Unit] =
    if WithAnimation.is4DOnlyTriangleMeshScene(newSpecs) then
      val slotsBuf: scala.collection.mutable.Map[Int, ArrayBuffer[Int]] =
        scala.collection.mutable.Map.empty
      val builder = TriangleMeshSceneBuilder(
        textureDir,
        mesh4DRecorder = (specIdx: Int, slotIdx: Int) =>
          slotsBuf.getOrElseUpdate(specIdx, ArrayBuffer.empty[Int]) += slotIdx
      )(using profilingConfig)
      val result = buildWithVideoTextureTracking(newSpecs):
        builder.buildScene(newSpecs, renderer, newSpecs.length)
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
      buildSceneTrackingVideoTextures(configs, newSpecs, renderer)

object WithAnimation:
  /** Per-mesh-slot bookkeeping for the GPU 4D projection animation fast path.
    * `slotsPerSpec(i)` lists the renderer slot indices that belong to spec i —
    * usually a single slot, but two for fractional 4D sponges (level n + level n+1). */
  final case class Anim4DState(specs: List[ObjectSpec], slotsPerSpec: Vector[Vector[Int]])
  final case class VideoTextureSlot(videoTexture: VideoTexture, textureIndices: Vector[Int])
  final case class VideoTextureState(specs: List[ObjectSpec], slots: Vector[VideoTextureSlot])
  final case class EnvMapVideoState(
    specs: List[ObjectSpec],
    envMapVideo: EnvMapVideo,
    textureIndex: Int
  )

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

  def specsCanReuseVideoTextureSlots(prev: List[ObjectSpec], next: List[ObjectSpec]): Boolean =
    prev == next && prev.exists(_.videoTexture.nonEmpty)

  def configsCanReuseEnvMapVideoSlot(
    prevSpecs: List[ObjectSpec],
    nextSpecs: List[ObjectSpec],
    prevEnvMapVideo: EnvMapVideo,
    nextEnvMapVideo: EnvMapVideo
  ): Boolean =
    prevSpecs == nextSpecs && prevEnvMapVideo.textureKey == nextEnvMapVideo.textureKey

  def hasVideoSources(
    configs: SceneConverter.SceneConfigs,
    specs: List[ObjectSpec]
  ): Boolean =
    specs.exists(_.videoTexture.nonEmpty) || configs.envMapVideo.nonEmpty
