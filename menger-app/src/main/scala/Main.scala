
import ch.qos.logback.classic.Level
import ch.qos.logback.classic.Logger
import com.badlogic.gdx.ApplicationListener
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration
import menger.MengerCLIOptions
import menger.ProfilingConfig
import menger.cli.PlaneConfig
import menger.common.Const
import menger.config.CameraConfig
import menger.config.EnvironmentConfig
import menger.config.ExecutionConfig
import menger.config.OptiXEngineConfig
import menger.config.SceneConfig
import menger.dsl.LoadedScene
import menger.dsl.SceneConverter
import menger.engines.AnimationEngine
import menger.engines.CliAnimationEngine
import menger.engines.InteractiveEngine
import menger.engines.PreviewEngine
import menger.engines.RenderEngine
import menger.engines.TAnimationConfig
import menger.engines.VideoEngine
import menger.optix.RenderConfig
import org.slf4j.LoggerFactory

object Main:
  def main(args: Array[String]): Unit =
    val opts = MengerCLIOptions(args.toList)
    configureLogging(opts.logLevel().toUpperCase)
    val config = getConfig(opts)
    try
      val rendering = createEngine(opts)
      rendering match
        case app: ApplicationListener => Lwjgl3Application(app, config)
        case _ => sys.error("Engine must implement ApplicationListener")
    catch
      case e: Exception =>
        System.err.println(s"Error: ${e.getMessage}")
        sys.exit(1)

  private def configureLogging(levelName: String): Unit =
    val level = Level.valueOf(levelName)
    // SLF4J returns the interface, but we need Logback's implementation to set the level
    LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME) match
      case logger: Logger =>
        logger.setLevel(level)
      case other =>
        val className = other.getClass.getName
        System.err.println(s"Warning: Expected Logback Logger but got $className")
        System.err.println("Logging level not set.")

  def getConfig(opts: MengerCLIOptions): Lwjgl3ApplicationConfiguration =
    val config = Lwjgl3ApplicationConfiguration()
    config.disableAudio(true)
    config.setTitle("Menger Sponges")
    config.setWindowedMode(opts.width(), opts.height())
    config.setBackBufferConfig(
      Const.Display.colorBits, Const.Display.colorBits,
      Const.Display.colorBits, Const.Display.colorBits,
      Const.Display.depthBits, Const.Display.stencilBits,
      opts.antialiasSamples()
    )
    // OptiX rendering requires fixed resolution for optimal performance
    config.setResizable(false)
    // Headless mode: render without displaying window (for CI/CD, batch processing)
    if opts.headless() then
      config.setInitialVisible(false)
      config.setDecorated(false)
    config

  def createEngine(opts: MengerCLIOptions): RenderEngine =
    given ProfilingConfig = opts.profileMinMs.toOption match
      case Some(minMs) => ProfilingConfig.enabled(minMs)
      case None => ProfilingConfig.disabled

    opts.scene.toOption match
      case Some(sceneName) => createSceneBasedEngine(opts, sceneName)
      case None => createCliBasedOptiXEngine(opts)

  private def createSceneBasedEngine(opts: MengerCLIOptions, sceneName: String)(using ProfilingConfig): RenderEngine =
    // Ensure all example scene objects are initialized so short names are registered
    val _ = examples.dsl.SceneIndex

    import menger.dsl.SceneLoader
    SceneLoader.load(sceneName) match
      case Right(LoadedScene.Animated(fn)) if opts.preview() =>
        val animConfig = TAnimationConfig(
          startT      = opts.startT(),
          endT        = opts.endT(),
          frames      = opts.tFrames.toOption.getOrElse(100),
          savePattern = ""
        )
        PreviewEngine(
          sceneFunction   = fn,
          previewConfig   = animConfig,
          executionConfig = buildExecutionConfig(opts),
          renderConfig    = opts.renderConfig,
          causticsConfig  = opts.causticsConfig
        )

      case Right(LoadedScene.Animated(fn)) if opts.tFrames.isSupplied =>
        // Multi-frame animation: create VideoEngine (with ffmpeg) or AnimationEngine (frames only)
        val animConfig = TAnimationConfig(
          startT = opts.startT(),
          endT = opts.endT(),
          frames = opts.tFrames(),
          savePattern = opts.saveName()
        )
        if opts.video.isSupplied then
          VideoEngine(
            sceneFunction = fn,
            animConfig = animConfig,
            executionConfig = buildExecutionConfig(opts),
            renderConfig = opts.renderConfig,
            causticsConfig = opts.causticsConfig,
            videoOutputPath = opts.video(),
            videoQuality = opts.videoQuality(),
            keepFrames = opts.keepFrames()
          )
        else
          AnimationEngine(
            sceneFunction = fn,
            animConfig = animConfig,
            executionConfig = buildExecutionConfig(opts),
            renderConfig = opts.renderConfig,
            causticsConfig = opts.causticsConfig
          )

      case Right(loadedScene) =>
        // Static scene or animated scene evaluated at fixed t
        val dslScene = loadedScene match
          case LoadedScene.Static(scene) => scene
          case LoadedScene.Animated(fn) =>
            fn(opts.freezeT.toOption.getOrElse(0f))
        createOptiXEngineFromDslScene(opts, dslScene)

      case Left(error) =>
        System.err.println(s"Failed to load scene '$sceneName': $error")
        sys.exit(1)

  private def createOptiXEngineFromDslScene(opts: MengerCLIOptions, dslScene: menger.dsl.Scene)(using ProfilingConfig): InteractiveEngine =
    val configs = SceneConverter.convert(dslScene, opts.causticsConfig)
    val baseRender = configs.render.getOrElse(RenderConfig.Default)
    val mergedRender = RenderConfig(
      shadows            = if opts.shadows.isSupplied            then opts.shadows()            else baseRender.shadows,
      transparentShadows = if opts.transparentShadows.isSupplied then opts.transparentShadows() else baseRender.transparentShadows,
      antialiasing       = if opts.antialiasing.isSupplied       then opts.antialiasing()       else baseRender.antialiasing,
      aaMaxDepth         = if opts.aaMaxDepth.isSupplied         then opts.aaMaxDepth()         else baseRender.aaMaxDepth,
      aaThreshold        = if opts.aaThreshold.isSupplied        then opts.aaThreshold()        else baseRender.aaThreshold,
    )
    val engineConfig = OptiXEngineConfig(
      scene = configs.scene,
      camera = configs.camera,
      environment = EnvironmentConfig(
        planes = configs.planes,
        lights = configs.lights,
        background = configs.background
      ),
      execution = buildExecutionConfig(opts),
      render = mergedRender,
      caustics = configs.caustics,
      cross = opts.crossConfig
    )
    InteractiveEngine(engineConfig, opts.userSetMaxInstances)

  private def createCliBasedOptiXEngine(opts: MengerCLIOptions)(using ProfilingConfig): RenderEngine =
    val engineConfig = OptiXEngineConfig(
      scene = SceneConfig(objectSpecs = opts.objects.toOption),
      camera = CameraConfig(
        position = opts.cameraPos(),
        lookAt = opts.cameraLookat(),
        up = opts.cameraUp()
      ),
      environment = EnvironmentConfig(
        planes = List(PlaneConfig(
          opts.plane(),
          opts.planeColor.toOption,
          opts.planeMaterial.toOption.flatMap(menger.optix.Material.fromName)
        )),
        lights = opts.light.toOption.getOrElse(List.empty),
        envMap = opts.envMap.toOption
      ),
      execution = buildExecutionConfig(opts),
      render = opts.renderConfig,
      caustics = opts.causticsConfig,
      cross = opts.crossConfig
    )
    opts.animate.toOption match
      case Some(animSpec) =>
        CliAnimationEngine(engineConfig, animSpec, opts.saveName())
      case None =>
        InteractiveEngine(engineConfig, opts.userSetMaxInstances)

  private def buildExecutionConfig(opts: MengerCLIOptions): ExecutionConfig =
    ExecutionConfig(
      fpsLogIntervalMs = opts.fpsLogInterval(),
      timeout = opts.timeout(),
      saveName = opts.saveName.toOption,
      enableStats = opts.stats() || opts.headless(),
      maxInstances = opts.maxInstances(),
      textureDir = opts.textureDir(),
      allowUniformRender = opts.allowUniformRender()
    )
