package menger

import scala.util.Try
import com.typesafe.scalalogging.LazyLogging

class AnimationSpecification(s: String) extends LazyLogging:

  type StartEnd = (Float, Float)

  val asMap: Option[Map[String, String]] =
    Try {s.split(":").map(_.split("=")).map(arr => (arr(0), arr(1))).toMap}.toOption

  val seconds: Option[Float] = asMap.flatMap(_.get("seconds")).flatMap(_.toFloatOption)

  val frames: Option[Int] = asMap.flatMap(_.get("frames")).flatMap(_.toIntOption)

  lazy val animationParameters: Map[String, StartEnd] =
    val parametersOnly = asMap.get -- AnimationSpecification.TIMESCALE_PARAMETERS
    parametersOnly.map { case (k, v) => k -> parseStartEnd(v) }

  def valid(spongeType: String): Boolean = timeSpecValid && animationParametersValid(spongeType)

  def rotationProjectionParameters(frame: Int): RotationProjectionParameters =
    def current(bounds: (Float, Float)): Float =
      bounds._1 + (bounds._2 - bounds._1) * frame / frames.get
    require(animationParameters.nonEmpty, "AnimationSpecification.animationParameters not defined")
    val rotXWBounds: (Float, Float) = animationParameters.get("rot-x-w").getOrElse(0f, 0f)
    val rotYWBounds: (Float, Float) = animationParameters.get("rot-y-w").getOrElse(0f, 0f)
    val rotZWBounds: (Float, Float) = animationParameters.get("rot-z-w").getOrElse(0f, 0f)
    val screenWBounds: (Float, Float) = animationParameters.get("projection-screen-w")
      .getOrElse(Const.defaultScreenW, Const.defaultScreenW)
    val eyeWBounds: (Float, Float) = animationParameters.get("projection-eye-w")
      .getOrElse(Const.defaultEyeW, Const.defaultEyeW)
    RotationProjectionParameters(
      rotXW = current(rotXWBounds),
      rotYW = current(rotYWBounds),
      rotZW = current(rotZWBounds),
      screenW = current(screenWBounds),
      eyeW = current(eyeWBounds)
    )

  private def parseStartEnd(s: String): StartEnd =
    val parts = s.split('-')
    require(parts.length == 2, s"Invalid start-end specification: $s")
    require(parts.head.toFloatOption.isDefined, s"Start ${parts.head} not a Float")
    require(parts.last.toFloatOption.isDefined, s"End ${parts.last} not a Float")
    (parts.head.toFloat, parts.last.toFloat)

  private def timeSpecValid: Boolean =
    (seconds.nonEmpty && seconds.get > 0) ^ (frames.nonEmpty && frames.get > 0)

  private def animationParametersValid(spongeType: String): Boolean =
    animationParameters.nonEmpty && 
      animationParameters.keySet.subsetOf(AnimationSpecification.validParameters(spongeType))

object AnimationSpecification:
  final val TIMESCALE_PARAMETERS = Set("frames", "seconds")
  final val ALWAYS_VALID_PARAMETERS = Set("rot-x", "rot-y", "rot-z")
  final val FOUR_D_VALID_PARAMETERS = Set(
    "rot-x-w", "rot-y-w", "rot-z-w", "projection-screen-w", "projection-eye-w"
  )
  final val FRACTAL_VALID_PARAMETERS = Set("level")
  final val FOUR_D_OBJECTS = Set("tesseract", "tesseract-sponge", "tesseract-sponge-2")
  final val FRACTAL_OBJECTS = Set("tesseract-sponge", "tesseract-sponge-2", "square", "cube")

  def validParameters(spongeType: String): Set[String] =
    ALWAYS_VALID_PARAMETERS ++ (
      if FOUR_D_OBJECTS.contains(spongeType) then FOUR_D_VALID_PARAMETERS else Set.empty) ++ (
      if FRACTAL_OBJECTS.contains(spongeType) then FRACTAL_VALID_PARAMETERS else Set.empty
    )


case class AnimationSpecifications(specification: List[String] = List.empty) extends LazyLogging:
  val parts: List[AnimationSpecification] = specification.map(AnimationSpecification(_))
  val numFrames: Int = parts.map(_.frames.getOrElse(0)).sum

  def valid(spongeType: String): Boolean =
    parts.forall(_.valid(spongeType)) && parts.map(_.seconds).map(_.isDefined).toSet.size < 2

  def rotationProjectionParameters(frame: Int): RotationProjectionParameters =
    require(parts.head.frames.isDefined, "AnimationSpecification.frames not defined")
    val partsBounds: Map[Int, (Int, Int)] =
      parts.zipWithIndex.map { case (part, i) =>
        i -> (0, part.frames.getOrElse(0))
      }.toMap
    logger.info(partsBounds.toString())
    ???

  def partAndFrame(totalFrame: Int): (Int, Int) =
    val part = parts.find(_.frames.exists(_ > totalFrame))
    part match
      case Some(p) =>
        val frame = totalFrame - p.frames.getOrElse(0)
        (parts.indexOf(p), frame)
      case None => (parts.length - 1, totalFrame - parts.last.frames.getOrElse(0))
