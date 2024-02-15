import org.rogach.scallop.*
import org.rogach.scallop.exceptions.ScallopException

class MengerCLIOptions(arguments: Seq[String]) extends ScallopConf(arguments):
  val timeout: ScallopOption[Float] = opt[Float](required = false, default = Some(0))
  val spongeType: ScallopOption[String] = choice(
    choices = List("box", "square"), required = false, default = Some("box")
  )
  val level: ScallopOption[Int] = opt[Int](required = false, default = Some(1), validate = _ >= 0)
  val lines: ScallopOption[Boolean] = opt[Boolean](required = false, default = Some(false))
  val width: ScallopOption[Int] = opt[Int](required = false, default = Some(800))
  val height: ScallopOption[Int] = opt[Int](required = false, default = Some(600))
  val antialiasSamples: ScallopOption[Int] = opt[Int](required = false, default = Some(4))
  verify()

  override def onError(e: Throwable): Unit = e match {
    case ScallopException(message) => throw IllegalArgumentException(message)
    case other => throw other
  }
