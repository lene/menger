package menger

import com.badlogic.gdx.backends.lwjgl3.{
  Lwjgl3Application, Lwjgl3ApplicationConfiguration
}
import org.rogach.scallop._

class MengerCLIConf(arguments: Seq[String]) extends ScallopConf(arguments):
  val timeout: ScallopOption[Float] = opt[Float](required = false, default = Some(0))
  val level: ScallopOption[Int] = opt[Int](required = false, default = Some(1), validate = _ >= 0)
  val lines: ScallopOption[Boolean] = opt[Boolean](required = false, default = Some(false))
  verify()

object Main:
  private final val COLOR_BITS = 8
  private final val DEPTH_BITS = 16
  private final val STENCIL_BITS = 0
  private final val NUM_ANTIALIAS_SAMPLES = 4

  def main(args: Array[String]): Unit =
    val conf = MengerCLIConf(args.toList)
    val config = Lwjgl3ApplicationConfiguration()
    config.disableAudio(true)
    config.setTitle("Engine Test")
    config.setWindowedMode(800, 600)
    config.setBackBufferConfig(
      COLOR_BITS, COLOR_BITS, COLOR_BITS, COLOR_BITS, DEPTH_BITS, STENCIL_BITS,
      NUM_ANTIALIAS_SAMPLES
    )
    Lwjgl3Application(EngineTest(conf.timeout(), conf.level(), conf.lines()), config)
