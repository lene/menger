import org.rogach.scallop._

class MengerCLIOptions(arguments: Seq[String]) extends ScallopConf(arguments):
  val timeout: ScallopOption[Float] = opt[Float](required = false, default = Some(0))
  val level: ScallopOption[Int] = opt[Int](required = false, default = Some(1), validate = _ >= 0)
  val lines: ScallopOption[Boolean] = opt[Boolean](required = false, default = Some(false))
  val width: ScallopOption[Int] = opt[Int](required = false, default = Some(800))
  val height: ScallopOption[Int] = opt[Int](required = false, default = Some(600))
  val antialias_samples: ScallopOption[Int] = opt[Int](required = false, default = Some(4))
  verify()
