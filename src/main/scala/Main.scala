package menger

import com.badlogic.gdx.backends.lwjgl3.{
  Lwjgl3Application, Lwjgl3ApplicationConfiguration
}
import org.rogach.scallop._

class MengerCLIOptions(arguments: Seq[String]) extends ScallopConf(arguments):
  val timeout: ScallopOption[Float] = opt[Float](required = false, default = Some(0))
  val level: ScallopOption[Int] = opt[Int](required = false, default = Some(1), validate = _ >= 0)
  val lines: ScallopOption[Boolean] = opt[Boolean](required = false, default = Some(false))
  val width: ScallopOption[Int] = opt[Int](required = false, default = Some(800))
  val height: ScallopOption[Int] = opt[Int](required = false, default = Some(600))
  val antialias_samples: ScallopOption[Int] = opt[Int](required = false, default = Some(4))
  verify()

object Main:
  private final val COLOR_BITS = 8
  private final val DEPTH_BITS = 16
  private final val STENCIL_BITS = 0

  def main(args: Array[String]): Unit =
    val opts = MengerCLIOptions(args.toList)
    val config = Lwjgl3ApplicationConfiguration()
    config.disableAudio(true)
    config.setTitle("Engine Test")
    config.setWindowedMode(opts.width(), opts.height())
    config.setBackBufferConfig(
      COLOR_BITS, COLOR_BITS, COLOR_BITS, COLOR_BITS, DEPTH_BITS, STENCIL_BITS,
      opts.antialias_samples()
    )
    Lwjgl3Application(EngineTest(opts.timeout(), opts.level(), opts.lines()), config)
