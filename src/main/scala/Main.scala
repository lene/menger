
import com.badlogic.gdx.backends.lwjgl3.{Lwjgl3Application, Lwjgl3ApplicationConfiguration}
import menger.{AnimatedMengerEngine, InteractiveMengerEngine, MengerCLIOptions, MengerEngine, RotationProjectionParameters}

object Main:
  private final val COLOR_BITS = 8
  private final val DEPTH_BITS = 16
  private final val STENCIL_BITS = 0

  def main(args: Array[String]): Unit =
    val opts = MengerCLIOptions(args.toList)
    val config = getConfig(opts)
    val rendering = createEngine(opts)
    Lwjgl3Application(rendering, config)

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
    val rotationProjectionParameters = RotationProjectionParameters(opts)
    if opts.animate().parts.nonEmpty then
      AnimatedMengerEngine(
        opts.spongeType(), opts.level(), rotationProjectionParameters, opts.lines(), opts.animate(),
        opts.saveName()
      )
    else
      InteractiveMengerEngine(
        opts.spongeType(), opts.level(), rotationProjectionParameters, opts.lines(), opts.timeout()
      )

