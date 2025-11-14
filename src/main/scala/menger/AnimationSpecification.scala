package menger

import scala.util.Try

import com.typesafe.scalalogging.LazyLogging

case class AnimationSpecification(s: String) extends LazyLogging:

  private type StartEnd = (start: Float, end: Float)

  val asMap: Option[Map[String, String]] =
    Try {s.split(":").map(_.split("=")).map(arr => (arr(0), arr(1))).toMap}.toOption

  val frames: Option[Int] = asMap.flatMap(_.get("frames")).flatMap(_.toIntOption)

  lazy val animationParameters: Map[String, StartEnd] =
    val parametersOnly = asMap.get -- AnimationSpecification.TIMESCALE_PARAMETERS
    parametersOnly.map { case (k, v) => k -> parseStartEnd(v) }

  def timeSpecValid: Boolean = frames.exists(_ > 0)
  def valid(spongeType: String): Boolean = timeSpecValid && animationParametersValid(spongeType)

  override def toString: String =
    // Safe: only called after AnimationSpecifications.require checks timeSpecValid
    val timeSpec = s"frames=${frames.get}"
    val animationSpec = animationParameters.mkString(":")
    s"$timeSpec:$animationSpec"

  private def current(bounds: StartEnd, frame: Int): Float =
    // Safe: only called after AnimationSpecifications.require checks timeSpecValid
    require(frame <= frames.get, s"Frame $frame exceeds total frames ${frames.get}")
    bounds.start + (bounds.end - bounds.start) * frame / frames.get

  def level(frame: Int): Option[Float] =
    animationParameters.get("level").map(current(_, frame))

  def rotationProjectionParameters(frame: Int): RotationProjectionParameters =
    require(animationParameters.nonEmpty, "AnimationSpecification.animationParameters not defined")
    val rotXBounds: StartEnd = animationParameters.get("rot-x").getOrElse(0f, 0f)
    val rotYBounds: StartEnd = animationParameters.get("rot-y").getOrElse(0f, 0f)
    val rotZBounds: StartEnd = animationParameters.get("rot-z").getOrElse(0f, 0f)
    val rotXWBounds: StartEnd = animationParameters.get("rot-x-w").getOrElse(0f, 0f)
    val rotYWBounds: StartEnd = animationParameters.get("rot-y-w").getOrElse(0f, 0f)
    val rotZWBounds: StartEnd = animationParameters.get("rot-z-w").getOrElse(0f, 0f)
    val screenWBounds: StartEnd = animationParameters.get("projection-screen-w")
      .getOrElse(Const.defaultScreenW, Const.defaultScreenW)
    val eyeWBounds: StartEnd = animationParameters.get("projection-eye-w")
      .getOrElse(Const.defaultEyeW, Const.defaultEyeW)
    RotationProjectionParameters(
      rotXW = current(rotXWBounds, frame),
      rotYW = current(rotYWBounds, frame),
      rotZW = current(rotZWBounds, frame),
      screenW = current(screenWBounds, frame),
      eyeW = current(eyeWBounds, frame),
      rotX = current(rotXBounds, frame),
      rotY = current(rotYBounds, frame),
      rotZ = current(rotZBounds, frame),
    )

  def isRotationAxisSet(x: Float, y: Float, z: Float, xw: Float, yw: Float, zw: Float): Boolean =
    (x != 0 && animationParameters.contains("rot-x")) ||
    (y != 0 && animationParameters.contains("rot-y")) ||
    (z != 0 && animationParameters.contains("rot-z")) ||
    (xw != 0 && animationParameters.contains("rot-x-w")) ||
    (yw != 0 && animationParameters.contains("rot-y-w")) ||
    (zw != 0 && animationParameters.contains("rot-z-w"))
    
  private def parseStartEnd(s: String): StartEnd =
    val parts = s.split('-')
    require(parts.length == 2, s"Invalid start-end specification: $s")
    require(parts.head.toFloatOption.isDefined, s"Start ${parts.head} not a Float")
    require(parts.last.toFloatOption.isDefined, s"End ${parts.last} not a Float")
    (parts.head.toFloat, parts.last.toFloat)

  private def animationParametersValid(spongeType: String): Boolean =
    animationParameters.nonEmpty && 
      animationParameters.keySet.subsetOf(AnimationSpecification.validParameters(spongeType))

object AnimationSpecification:
  final val TIMESCALE_PARAMETERS = Set("frames")
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


