
import ch.qos.logback.classic.Level
import ch.qos.logback.classic.Logger
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration
import menger.MengerCLIOptions
import menger.ProfilingConfig
import menger.RotationProjectionParameters
import menger.common.Const
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
    config

  def createEngine(opts: MengerCLIOptions): RenderEngine =
    given ProfilingConfig = opts.profileMinMs.toOption match
      case Some(minMs) => ProfilingConfig.enabled(minMs)
      case None => ProfilingConfig.disabled

    val rotationProjectionParameters = RotationProjectionParameters(opts)

    if opts.optix() then createOptiXEngine(opts, rotationProjectionParameters)
    else if opts.animate.toOption.exists(_.parts.nonEmpty) then
      createAnimatedEngine(opts, rotationProjectionParameters)
    else createInteractiveEngine(opts, rotationProjectionParameters)

  private def createOptiXEngine(
    opts: MengerCLIOptions,
    rotationProjectionParams: RotationProjectionParameters
  )(using ProfilingConfig): OptiXEngine =
    // --object or --objects is required for OptiX (validated in MengerCLIOptions)
    OptiXEngine(
      opts.objectType.toOption.getOrElse("sphere"), // Legacy default
      opts.level(), opts.lines(), opts.color(),
      opts.fpsLogInterval(),
      opts.radius(), opts.ior(), opts.scale(),
      opts.cameraPos(), opts.cameraLookat(), opts.cameraUp(), opts.center(), opts.plane(),
      opts.planeColor.toOption,
      opts.timeout(),
      opts.saveName.toOption,
      opts.stats(),
      opts.light.toOption,
      opts.renderConfig,
      opts.causticsConfig,
      opts.maxInstances(),
      opts.objects.toOption
    )

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

