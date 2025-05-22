package menger

import scala.util.Try
import com.typesafe.scalalogging.LazyLogging

case class AnimationSpecification(s: String) extends LazyLogging:

  type StartEnd = (Float, Float)

  val asMap: Option[Map[String, String]] =
    Try {s.split(":").map(_.split("=")).map(arr => (arr(0), arr(1))).toMap}.toOption

  val frames: Option[Int] = asMap.flatMap(_.get("frames")).flatMap(_.toIntOption)

  lazy val animationParameters: Map[String, StartEnd] =
    val parametersOnly = asMap.get -- AnimationSpecification.TIMESCALE_PARAMETERS
    parametersOnly.map { case (k, v) => k -> parseStartEnd(v) }

  def timeSpecValid: Boolean = frames.nonEmpty && frames.get > 0
  def valid(spongeType: String): Boolean = timeSpecValid && animationParametersValid(spongeType)

  override def toString: String =
    val timeSpec = s"frames=${frames.get}"
    val animationSpec = animationParameters.mkString(":")
    s"$timeSpec:$animationSpec"

  def rotationProjectionParameters(frame: Int): RotationProjectionParameters =
    def current(bounds: (Float, Float)): Float =
      bounds._1 + (bounds._2 - bounds._1) * frame / frames.get

    require(animationParameters.nonEmpty, "AnimationSpecification.animationParameters not defined")
    require(frame <= frames.get, s"Frame $frame exceeds total frames ${frames.get}")
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
      rotXW = current(rotXWBounds),
      rotYW = current(rotYWBounds),
      rotZW = current(rotZWBounds),
      screenW = current(screenWBounds),
      eyeW = current(eyeWBounds),
      rotX = current(rotXBounds),
      rotY = current(rotYBounds),
      rotZ = current(rotZBounds),
    )

  def isRotationAxisSet(x: Float, y: Float, z: Float, xw: Float, yw: Float, zw: Float): Boolean =
    x != 0 && animationParameters.contains("rot-x") ||
    y != 0 && animationParameters.contains("rot-y") ||
    z != 0 && animationParameters.contains("rot-z") ||
    xw != 0 && animationParameters.contains("rot-x-w") ||
    yw != 0 && animationParameters.contains("rot-y-w") ||
    zw != 0 && animationParameters.contains("rot-z-w")
    
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


