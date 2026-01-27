
import ch.qos.logback.classic.Level
import ch.qos.logback.classic.Logger
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration
import menger.MengerCLIOptions
import menger.ProfilingConfig
import menger.RotationProjectionParameters
import menger.common.Const
import menger.config.CameraConfig
import menger.config.EnvironmentConfig
import menger.config.ExecutionConfig
import menger.config.OptiXEngineConfig
import menger.config.SceneConfig
import menger.engines.AnimatedMengerEngine
import menger.engines.InteractiveMengerEngine
import menger.engines.OptiXEngine
import menger.engines.RenderEngine
import org.slf4j.LoggerFactory

object Main:
  def main(args: Array[String]): Unit =
    val opts = MengerCLIOptions(args.toList)
    configureLogging(opts.logLevel().toUpperCase)
    val config = getConfig(opts)
    try
      val rendering = createEngine(opts)
      Lwjgl3Application(rendering, config)
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
    // Disable window resizing for OptiX mode to prevent expensive buffer reallocation
    // OptiX rendering requires fixed resolution for optimal performance
    if opts.optix() then config.setResizable(false)
    // Headless mode: render without displaying window (for CI/CD, batch processing)
    if opts.headless() then
      config.setInitialVisible(false)
      config.setDecorated(false)
    config

  def createEngine(opts: MengerCLIOptions): RenderEngine =
    given ProfilingConfig = opts.profileMinMs.toOption match
      case Some(minMs) => ProfilingConfig.enabled(minMs)
      case None => ProfilingConfig.disabled

    val rotationProjectionParameters = RotationProjectionParameters(opts)

    if opts.optix() then createOptiXEngine(opts)
    else if opts.animate.toOption.exists(_.parts.nonEmpty) then
      createAnimatedEngine(opts, rotationProjectionParameters)
    else createInteractiveEngine(opts, rotationProjectionParameters)

  private def createOptiXEngine(opts: MengerCLIOptions)(using ProfilingConfig): OptiXEngine =
    val engineConfig = OptiXEngineConfig(
      scene = SceneConfig(
        objectSpecs = opts.objects.toOption
      ),
      camera = CameraConfig(
        position = opts.cameraPos(),
        lookAt = opts.cameraLookat(),
        up = opts.cameraUp()
      ),
      environment = EnvironmentConfig(
        plane = opts.plane(),
        planeColor = opts.planeColor.toOption,
        lights = opts.light.toOption.getOrElse(List.empty)
      ),
      execution = ExecutionConfig(
        fpsLogIntervalMs = opts.fpsLogInterval(),
        timeout = opts.timeout(),
        saveName = opts.saveName.toOption,
        enableStats = opts.stats(),
        maxInstances = opts.maxInstances(),
        textureDir = opts.textureDir()
      ),
      render = opts.renderConfig,
      caustics = opts.causticsConfig
    )
    OptiXEngine(engineConfig, opts.userSetMaxInstances)

  private def createAnimatedEngine(
    opts: MengerCLIOptions,
    rotationProjectionParams: RotationProjectionParameters
  )(using ProfilingConfig): AnimatedMengerEngine =
    AnimatedMengerEngine(
      opts.spongeType(), opts.level(), rotationProjectionParams, opts.lines(), opts.color(),
      opts.animate(), opts.saveName.toOption, opts.faceColor.toOption, opts.lineColor.toOption,
      opts.fpsLogInterval()
    )

  private def createInteractiveEngine(
    opts: MengerCLIOptions,
    rotationProjectionParams: RotationProjectionParameters
  )(using ProfilingConfig): InteractiveMengerEngine =
    InteractiveMengerEngine(
      opts.spongeType(), opts.level(), rotationProjectionParams, opts.lines(), opts.color(),
      opts.timeout(), opts.faceColor.toOption, opts.lineColor.toOption,
      opts.fpsLogInterval()
    )
