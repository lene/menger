package menger

import com.typesafe.scalalogging.LazyLogging
import org.rogach.scallop.*
import org.rogach.scallop.exceptions.ScallopException

import scala.util.Try

class MengerCLIOptions(arguments: Seq[String]) extends ScallopConf(arguments) with LazyLogging:
  version("menger v0.2.6 (c) 2023-25, lene.preuss@gmail.com")
  val timeout: ScallopOption[Float] = opt[Float](required = false, default = Some(0))
  val spongeType: ScallopOption[String] = choice(
    choices = List("cube", "square", "tesseract", "tesseract-sponge", "tesseract-sponge-2"), 
    required = false, default = Some("square")
  )
  val projectionScreenW: ScallopOption[Float] = opt[Float](
    required = false, default = Some(Const.defaultScreenW), validate = _ > 0
  )
  val projectionEyeW: ScallopOption[Float] = opt[Float](
    required = false, default = Some(Const.defaultEyeW), validate = _ > 0
  )
  private def degreeOpt = opt[Float](
      required = false, default = Some(0), validate = a => a >= 0 && a < 360
    )
  val rotX: ScallopOption[Float] = degreeOpt
  val rotY: ScallopOption[Float] = degreeOpt
  val rotZ: ScallopOption[Float] = degreeOpt
  val rotXW: ScallopOption[Float] = degreeOpt
  val rotYW: ScallopOption[Float] = degreeOpt
  val rotZW: ScallopOption[Float] = degreeOpt
  val level: ScallopOption[Int] = opt[Int](required = false, default = Some(1), validate = _ >= 0)
  val lines: ScallopOption[Boolean] = opt[Boolean](required = false, default = Some(false))
  val width: ScallopOption[Int] = opt[Int](
    required = false, default = Some(Const.defaultWindowWidth)
  )
  val height: ScallopOption[Int] = opt[Int](
    required = false, default = Some(Const.defaultWindowHeight)
  )
  val antialiasSamples: ScallopOption[Int] = opt[Int](
    required = false, default = Some(Const.defaultAntialiasSamples)
  )
  val animate: ScallopOption[AnimationSpecifications] = opt[AnimationSpecifications]()(
    using animationSpecificationsConverter
  )
  val saveName: ScallopOption[String] = opt[String](
    required = false,  validate = _.nonEmpty
  )

  mutuallyExclusive(timeout, animate)
  validate(projectionScreenW, projectionEyeW) { (screen, eye) =>
    if eye > screen then Right(())
    else Left("eyeW must be greater than screenW")
  }
  validateOpt(animate, spongeType) {
    case (Some(spec), Some(sponge)) => validateAnimationSpecification(spec, sponge)
    case _ => Right(())
  }

  validateOpt(animate, rotX, rotY, rotZ, rotXW, rotYW, rotZW) { (spec, x, y, z, xw, yw, zw) =>
    if spec.isEmpty then Right(())
    else
    if !spec.get.isRotationAxisSet(
      x.getOrElse(0), y.getOrElse(0), z.getOrElse(0), xw.getOrElse(0), yw.getOrElse(0), zw.getOrElse(0)
    ) then Right(())
    else Left("Animation specification has rotation axis set that is also set statically")
  }

  verify()

  private def validateAnimationSpecification(spec: AnimationSpecifications, spongeType: String) =
    if spec.valid(spongeType) && spec.timeSpecValid then Right(())
    else Left("Invalid animation specification")


val animationSpecificationsConverter = new ValueConverter[AnimationSpecifications] {
  val argType: ArgType.V = org.rogach.scallop.ArgType.LIST
  def parse(s: List[(String, List[String])]): Either[String, Option[AnimationSpecifications]] =
    val specStrings = s.flatMap(_(1))
    if specStrings.isEmpty then Right(None)
    else
      Try { Right(Some(AnimationSpecifications(specStrings)))
      }.recover { case e: Exception => Left(e.getMessage) }.get
}
