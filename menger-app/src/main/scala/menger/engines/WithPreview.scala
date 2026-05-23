package menger.engines

import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.graphics.GL20
import com.typesafe.scalalogging.LazyLogging
import menger.Vector3Extensions.toVector3
import menger.common.ImageSize
import menger.dsl.Scene
import menger.input.GdxRuntime
import menger.optix.CausticsConfig
import menger.optix.RenderConfig

trait WithPreview extends RenderEngine with LazyLogging:
  self: BaseEngine =>

  protected def sceneFunction: Float => Scene
  protected def previewConfig: TAnimationConfig
  protected def renderConfig: RenderConfig
  protected def causticsConfig: CausticsConfig
  protected def firstFrameConfigs: SceneConverter.SceneConfigs
  protected def windowTitle: String = "Menger Sponges"

  private val currentT    = new AtomicReference[Float](0f)
  private val isPlaying   = new AtomicBoolean(false)
  private val needsRender = new AtomicBoolean(true)

  private def tStep: Float =
    val range = previewConfig.endT - previewConfig.startT
    if previewConfig.frames > 1 then range / (previewConfig.frames - 1) else range

  def stepT(delta: Float): Unit =
    val clamped = clampT(currentT.get() + delta)
    currentT.set(clamped)
    updateTitle()
    needsRender.set(true)
    GdxRuntime.requestRendering()

  def jumpToStart(): Unit =
    currentT.set(previewConfig.startT)
    updateTitle()
    needsRender.set(true)
    GdxRuntime.requestRendering()

  def jumpToEnd(): Unit =
    currentT.set(previewConfig.endT)
    updateTitle()
    needsRender.set(true)
    GdxRuntime.requestRendering()

  def togglePlay(): Unit =
    val nowPlaying = !isPlaying.get()
    isPlaying.set(nowPlaying)
    GdxRuntime.setContinuousRendering(nowPlaying)
    if nowPlaying then GdxRuntime.requestRendering()

  private def clampT(v: Float): Float =
    math.max(previewConfig.startT, math.min(previewConfig.endT, v))

  private def frameForT(t: Float): Int =
    if tStep > 0 then math.round((t - previewConfig.startT) / tStep) else 0

  private def updateTitle(): Unit =
    val t     = currentT.get()
    val frame = frameForT(t)
    GdxRuntime.setWindowTitle(
      f"$windowTitle | t=$t%.3f | frame $frame/${previewConfig.frames}"
    )

  abstract override def create(): Unit =
    currentT.set(previewConfig.startT)
    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureLights(renderer)
    sceneConfigurator.configureCamera(renderer)
    buildSceneFromConfigs(firstFrameConfigs, renderer).recover { case e: Exception =>
      logger.error(s"Failed to create initial preview scene: ${e.getMessage}", e)
      GdxRuntime.exit()
    }.get
    renderer.setRenderConfig(renderConfig)
    renderer.setCausticsConfig(firstFrameConfigs.caustics)
    PlaneConfigurer.configurePlanes(renderer, firstFrameConfigs.planes.toArray)
    GdxRuntime.setContinuousRendering(false)
    updateTitle()

  abstract override def render(): Unit =
    GdxRuntime.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)
    val width  = GdxRuntime.width
    val height = GdxRuntime.height

    if isPlaying.get() then
      val next = currentT.get() + tStep
      if next >= previewConfig.endT then
        currentT.set(previewConfig.endT)
        togglePlay()
      else
        currentT.set(next)
      updateTitle()
      needsRender.set(true)

    if needsRender.getAndSet(false) && width > 0 && height > 0 then
      val t = currentT.get()
      Try(sceneFunction(t)) match
        case Failure(e) =>
          logger.error(s"Scene function threw for t=$t: ${e.getMessage}", e)
        case scala.util.Success(dslScene) =>
          val configs  = SceneConverter.convert(dslScene, causticsConfig)
          val renderer = rendererWrapper.renderer
          renderer.clearAllInstances()
          buildSceneFromConfigs(configs, renderer).recover { case e: Exception =>
            logger.error(s"Failed to build preview scene for t=$t: ${e.getMessage}", e)
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
