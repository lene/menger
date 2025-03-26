package menger

import org.rogach.scallop.*
import org.rogach.scallop.exceptions.ScallopException

import scala.util.Try

class MengerCLIOptions(arguments: Seq[String]) extends ScallopConf(arguments):
  val timeout: ScallopOption[Float] = opt[Float](required = false, default = Some(0))
  val spongeType: ScallopOption[String] = choice(
    choices = List("cube", "square", "tesseract", "tesseract-sponge", "tesseract-sponge-2"), 
    required = false, default = Some("square")
  )
  val projectionScreenW: ScallopOption[Float] = opt[Float](
    required = false, default = Some(1), validate = _ > 0
  )
  val projectionEyeW: ScallopOption[Float] = opt[Float](
    required = false, default = Some(2), validate = _ > 0
  )
  private def degreeOpt = opt[Float](
      required = false, default = Some(0), validate = a => a >= 0 && a < 360
    )
  val rotXW: ScallopOption[Float] = degreeOpt
  val rotYW: ScallopOption[Float] = degreeOpt
  val rotZW: ScallopOption[Float] = degreeOpt
  val level: ScallopOption[Int] = opt[Int](required = false, default = Some(1), validate = _ >= 0)
  val lines: ScallopOption[Boolean] = opt[Boolean](required = false, default = Some(false))
  val width: ScallopOption[Int] = opt[Int](required = false, default = Some(800))
  val height: ScallopOption[Int] = opt[Int](required = false, default = Some(600))
  val antialiasSamples: ScallopOption[Int] = opt[Int](required = false, default = Some(4))
  // TODO: use custom converters for AnimationSpecification, see https://github.com/scallop/scallop/wiki/Custom-converters
  val animate: ScallopOption[AnimationSpecifications] = opt[AnimationSpecifications]()(animationSpecificationsConverter)
  validate(projectionScreenW, projectionEyeW) { (screen, eye) =>
    if eye > screen then Right(())
    else Left("eyeW must be greater than screenW")
  }
  validate(animate, spongeType) { (start, sponge) => validateAnimationSpecification(start, sponge) }
  verify()

  private def validateAnimationSpecification(spec: AnimationSpecifications, spongeType: String) =
    if spec.valid(spongeType) then Right(())
    else Left("Invalid animation specification")

  override def onError(e: Throwable): Unit = e match
    case ScallopException(message) => throw IllegalArgumentException(message)
    case other => throw other

val animationSpecificationsConverter: ValueConverter[AnimationSpecifications] = new ValueConverter[AnimationSpecifications] {
  val argType = org.rogach.scallop.ArgType.LIST
  def parse(s: List[(String, List[String])]): Either[String, Option[AnimationSpecifications]] =
    val specificationStrings = s.flatMap(_(1))
    Try {
      Right(Some(AnimationSpecifications(specificationStrings)))
    }.recover {
      case e: Exception => Left(e.getMessage)
    }.get
}
