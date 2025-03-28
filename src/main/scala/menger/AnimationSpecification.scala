package menger

import scala.util.Try
import com.typesafe.scalalogging.LazyLogging

case class AnimationSpecification(s: String) extends LazyLogging:

  type StartEnd = (Float, Float)

  val asMap: Option[Map[String, String]] =
    Try {s.split(":").map(_.split("=")).map(arr => (arr(0), arr(1))).toMap}.toOption

  val seconds: Option[Float] = asMap.flatMap(_.get("seconds")).flatMap(_.toFloatOption)

  val frames: Option[Int] = asMap.flatMap(_.get("frames")).flatMap(_.toIntOption)

  lazy val animationParameters: Map[String, StartEnd] =
    val parametersOnly = asMap.get -- AnimationSpecification.TIMESCALE_PARAMETERS
    parametersOnly.map { case (k, v) => k -> parseStartEnd(v) }

  def valid(spongeType: String): Boolean = timeSpecValid && animationParametersValid(spongeType)

  override def toString: String =
    val timeSpec = (seconds, frames) match
      case (Some(s), None) => s"seconds=$s"
      case (None, Some(f)) => s"frames=$f"
      case _ => throw IllegalArgumentException("Invalid animation specification")
    val animationSpec = animationParameters.mkString(":")
    s"$timeSpec:$animationSpec"

  def rotationProjectionParameters(frame: Int): RotationProjectionParameters =
    def current(bounds: (Float, Float)): Float = {
      bounds._1 + (bounds._2 - bounds._1) * frame / frames.get
    }

    require(animationParameters.nonEmpty, "AnimationSpecification.animationParameters not defined")
    val rotXBounds: (Float, Float) = animationParameters.get("rot-x").getOrElse(0f, 0f)
    val rotYBounds: (Float, Float) = animationParameters.get("rot-y").getOrElse(0f, 0f)
    val rotZBounds: (Float, Float) = animationParameters.get("rot-z").getOrElse(0f, 0f)
    val rotXWBounds: (Float, Float) = animationParameters.get("rot-x-w").getOrElse(0f, 0f)
    val rotYWBounds: (Float, Float) = animationParameters.get("rot-y-w").getOrElse(0f, 0f)
    val rotZWBounds: (Float, Float) = animationParameters.get("rot-z-w").getOrElse(0f, 0f)
    val screenWBounds: (Float, Float) = animationParameters.get("projection-screen-w")
      .getOrElse(Const.defaultScreenW, Const.defaultScreenW)
    val eyeWBounds: (Float, Float) = animationParameters.get("projection-eye-w")
      .getOrElse(Const.defaultEyeW, Const.defaultEyeW)
    RotationProjectionParameters(
      rotX = current(rotXBounds),
      rotY = current(rotYBounds),
      rotZ = current(rotZBounds),
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
    partAndFrame(frame).map((specList, frame) => accumulateAllButLastRotationProjections(specList) + specList.last.rotationProjectionParameters(frame)).getOrElse(
      throw IllegalArgumentException("AnimationSpecification.frames not defined")
    )

  def accumulateAllButLastRotationProjections(specs: List[AnimationSpecification]): RotationProjectionParameters =
    specs.init.foldLeft(RotationProjectionParameters()) { (acc, spec) =>
      acc + spec.rotationProjectionParameters(spec.frames.getOrElse(0))
    }

  def partAndFrame(
    totalFrame: Int,
    partsParts: List[AnimationSpecification] = parts,
    accumulator: List[AnimationSpecification] = List.empty
  ): Try[(List[AnimationSpecification], Int)] =
    if partsParts.isEmpty then
       throw IllegalArgumentException("AnimationSpecification.parts not defined")
    else
      val current = partsParts.head
      if current.frames.isEmpty then
        throw IllegalArgumentException(s"Animation specification $current has no frames")
      else
        if current.frames.getOrElse(0) > totalFrame then
          Try((accumulator :+ current, totalFrame))
        else partAndFrame(totalFrame - current.frames.getOrElse(0), partsParts.tail, accumulator :+ current)
