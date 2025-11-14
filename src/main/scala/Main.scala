
import ch.qos.logback.classic.Level
import ch.qos.logback.classic.Logger
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration
import menger.MengerCLIOptions
import menger.ProfilingConfig
import menger.RotationProjectionParameters
import menger.engines.AnimatedMengerEngine
import menger.engines.InteractiveMengerEngine
import menger.engines.MengerEngine
import menger.engines.OptiXEngine
import org.slf4j.LoggerFactory

object Main:
  private final val COLOR_BITS = 8
  private final val DEPTH_BITS = 16
  private final val STENCIL_BITS = 0

  def main(args: Array[String]): Unit =
    val opts = MengerCLIOptions(args.toList)
    configureLogging(opts.logLevel().toUpperCase)
    val config = getConfig(opts)
    val rendering = createEngine(opts)
    Lwjgl3Application(rendering, config)

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
      COLOR_BITS, COLOR_BITS, COLOR_BITS, COLOR_BITS, DEPTH_BITS, STENCIL_BITS,
      opts.antialiasSamples()
    )
    config

  def createEngine(opts: MengerCLIOptions): MengerEngine =
    given ProfilingConfig = opts.profileMinMs.toOption match
      case Some(minMs) => ProfilingConfig.enabled(minMs)
      case None => ProfilingConfig.disabled

    val rotationProjectionParameters = RotationProjectionParameters(opts)

    if opts.optix() then createOptiXEngine(opts, rotationProjectionParameters)
    else if opts.animate.isDefined && opts.animate().parts.nonEmpty then
      createAnimatedEngine(opts, rotationProjectionParameters)
    else createInteractiveEngine(opts, rotationProjectionParameters)

  private def createOptiXEngine(
    opts: MengerCLIOptions,
    rpp: RotationProjectionParameters
  )(using ProfilingConfig): OptiXEngine =
    OptiXEngine(
      opts.spongeType(), opts.level(), rpp, opts.lines(), opts.color(),
      opts.faceColor.toOption, opts.lineColor.toOption,
      opts.fpsLogInterval(),
      opts.radius(), opts.ior(), opts.scale(),
      opts.cameraPos(), opts.cameraLookat(), opts.cameraUp(), opts.center(), opts.plane(),
      opts.timeout(),
      opts.saveName.toOption,
      opts.stats()
    )

  private def createAnimatedEngine(
    opts: MengerCLIOptions,
    rpp: RotationProjectionParameters
  )(using ProfilingConfig): AnimatedMengerEngine =
    AnimatedMengerEngine(
      opts.spongeType(), opts.level(), rpp, opts.lines(), opts.color(),
      opts.animate(), opts.saveName.toOption, opts.faceColor.toOption, opts.lineColor.toOption,
      opts.fpsLogInterval()
    )

  private def createInteractiveEngine(
    opts: MengerCLIOptions,
    rpp: RotationProjectionParameters
  )(using ProfilingConfig): InteractiveMengerEngine =
    InteractiveMengerEngine(
      opts.spongeType(), opts.level(), rpp, opts.lines(), opts.color(),
      opts.timeout(), opts.faceColor.toOption, opts.lineColor.toOption,
      opts.fpsLogInterval()
    )

