package menger

import scala.util.Try
import com.typesafe.scalalogging.LazyLogging

class AnimationSpecification(s: String, spongeType: String) extends LazyLogging:

  type StartEnd = (Float, Float)

  val asMap: Option[Map[String, String]] =
    Try {s.split(":").map(_.split("=")).map(arr => (arr(0), arr(1))).toMap}.toOption

  val seconds: Option[Float] = asMap.flatMap(_.get("seconds")).flatMap(_.toFloatOption)

  val frames: Option[Int] = asMap.flatMap(_.get("frames")).flatMap(_.toIntOption)

  lazy val animationParameters: Map[String, StartEnd] =
    val parametersOnly = asMap.get -- AnimationSpecification.TIMESCALE_PARAMETERS
    parametersOnly.map { case (k, v) => k -> parseStartEnd(v) }

  def valid: Boolean = timeSpecValid && animationParametersValid

  private def parseStartEnd(s: String): StartEnd =
    val splitted = s.split('-')
    require(splitted.length == 2, s"Invalid start-end specification: $s")
    require(splitted.head.toFloatOption.isDefined, s"Start ${splitted.head} not a Float")
    require(splitted.last.toFloatOption.isDefined, s"End ${splitted.last} not a Float")
    (splitted.head.toFloat, splitted.last.toFloat)

  private def timeSpecValid: Boolean =
    (seconds.nonEmpty && seconds.get > 0) ^ (frames.nonEmpty && frames.get > 0)

  private def animationParametersValid: Boolean =
    val validParameters = AnimationSpecification.ALWAYS_VALID_PARAMETERS ++ (spongeType match
      case "tesseract" => AnimationSpecification.FOUR_D_VALID_PARAMETERS
      case "tesseract-sponge" | "tesseract-sponge-2" => AnimationSpecification.FOUR_D_VALID_PARAMETERS ++ AnimationSpecification.FRACTAL_VALID_PARAMETERS
      case "square" | "cube" => AnimationSpecification.FRACTAL_VALID_PARAMETERS
      case _ => Seq.empty
    )
    logger.warn(s"Valid parameters for $spongeType: $validParameters")
    animationParameters.nonEmpty && animationParameters.keySet.subsetOf(validParameters)

object AnimationSpecification:
  final val TIMESCALE_PARAMETERS = Set("frames", "seconds")
  final val ALWAYS_VALID_PARAMETERS = Set("rot-x", "rot-y", "rot-z")
  final val FOUR_D_VALID_PARAMETERS = Set("rot-x-w", "rot-y-w", "rot-z-w", "projection-screen-w", "projection-eye-w")
  final val FRACTAL_VALID_PARAMETERS = Set("level")


class AnimationSpecifications(specification: List[String], spongeType: String):
  private val parts: List[AnimationSpecification] = specification.map(AnimationSpecification(_, spongeType))
  def valid: Boolean = parts.forall(_.valid)