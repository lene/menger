package menger

import scala.util.Try

import com.badlogic.gdx.graphics.Color
import com.typesafe.scalalogging.LazyLogging
import org.rogach.scallop.*

class MengerCLIOptions(arguments: Seq[String]) extends ScallopConf(arguments) with LazyLogging:
  version("menger v0.2.8 (c) 2023-25, lene.preuss@gmail.com")

  private def validateSpongeType(spongeType: String): Boolean =
    isValidSpongeType(spongeType)

  private val basicSpongeTypes = List("cube", "square", "square-sponge", "cube-sponge", "tesseract", "tesseract-sponge", "tesseract-sponge-2")
  private val compositePattern = """composite\[(.+)]""".r

  private def isValidSpongeType(spongeType: String): Boolean =
    if basicSpongeTypes.contains(spongeType) then true
    else spongeType match
      case compositePattern(content) =>
        // Only allow cube and square in composites (no nesting, no whitespace)
        val components = content.split(",").toSet
        val allowed = Set("cube", "square")
        components.nonEmpty && components.subsetOf(allowed)
      case _ => false


  val timeout: ScallopOption[Float] = opt[Float](required = false, default = Some(0))
  val spongeType: ScallopOption[String] = opt[String](
    required = false, default = Some("square"),
    validate = validateSpongeType
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
  val level: ScallopOption[Float] = opt[Float](required = false, default = Some(1.0f), validate = _ >= 0)
  val lines: ScallopOption[Boolean] = opt[Boolean](required = false, default = Some(false))
  val color: ScallopOption[Color] = opt[Color](required = false, default = Some(Color.LIGHT_GRAY))(
    using colorConverter
  )
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

val colorConverter = new ValueConverter[Color] {
  val argType: ArgType.V = org.rogach.scallop.ArgType.SINGLE
  def parse(s: List[(String, List[String])]): Either[String, Option[Color]] =
    if s.isEmpty || s.head._2.isEmpty then Right(None)
    else
      val input = s.head._2.head.trim
      Try { doParse(input) }.recover {
        case e: Exception => Left(s"Color '$input' not recognized: ${e.getMessage}")
      }.get

  private def doParse(input: String): Either[String, Option[Color]] =
    if input.contains(',') then parseInts(input)
    else parseHex(input)

  private def parseHex(input: String): Either[String, Option[Color]] =
    input.length match
      case len if len >= 6 && len <= 8 => Right(Some(Color.valueOf(input)))
      case _ => Left(s"Color '$input' must be a name or a hex value RRGGBB or RRGGBBAA")

  private def parseInts(input: String): Either[String, Option[Color]] =
    val parts = input.trim.split(",").map(_.trim)
    parts.length match
      case n if input.startsWith(",") || input.endsWith(",") =>
        Left(s"Color '$input' must not start or end with a comma")
      case n if n < 3 || n > 4 =>
        Left(s"Color '$input' must have 3 or 4 components")
      case _ =>
        val nums = parts.map(_.toInt)
        if nums.exists(n => n < 0 || n > 255) then
            Left(s"Color '$input' has values out of range 0-255")
        else
          val Array(r, g, b, a) = nums.map(_ / 255f).padTo(4, 1f)
          Right(Some(Color(r, g, b, a)))
}