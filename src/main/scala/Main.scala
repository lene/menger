
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
      case logger: Logger => logger.setLevel(level)
      case _ => // Should never happen when Logback is the SLF4J implementation

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
    if opts.optix() then
      OptiXEngine(
        opts.spongeType(), opts.level(), rotationProjectionParameters, opts.lines(), opts.color(),
        opts.faceColor.toOption, opts.lineColor.toOption,
        opts.fpsLogInterval(),
        opts.radius(), opts.ior(), opts.timeout(),
        opts.saveName.toOption
      )
    else if opts.animate.isDefined && opts.animate().parts.nonEmpty then
      AnimatedMengerEngine(
        opts.spongeType(), opts.level(), rotationProjectionParameters, opts.lines(), opts.color(),
        opts.animate(), opts.saveName.toOption, opts.faceColor.toOption, opts.lineColor.toOption,
        opts.fpsLogInterval()
      )
    else
      InteractiveMengerEngine(
        opts.spongeType(), opts.level(), rotationProjectionParameters, opts.lines(), opts.color(),
        opts.timeout(), opts.faceColor.toOption, opts.lineColor.toOption,
        opts.fpsLogInterval()
      )

