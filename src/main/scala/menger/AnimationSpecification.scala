package menger

import scala.util.Try

import com.typesafe.scalalogging.LazyLogging
import menger.common.Const

case class AnimationSpecification(specString: String) extends LazyLogging:

  private type StartEnd = (start: Float, end: Float)

  /**
   * Parses animation specification string into key-value map.
   * 
   * Format: "key1=value1:key2=value2:..."
   * 
   * Example input: "frames=5:rot-y=0-90:level=0-2"
   * Example output: Map("frames" -> "5", "rot-y" -> "0-90", "level" -> "0-2")
   * 
   * @param specString Animation specification in colon-separated key=value format
   * @return Map of parameter names to their string values, or empty Map if parsing fails
   */
  private def parseSpecString(specString: String): Map[String, String] =
    Try {
      specString
        .split(":")               // Split by colon: ["frames=5", "rot-y=0-90", ...]
        .map(_.split("="))        // Split each by equals: [["frames", "5"], ["rot-y", "0-90"], ...]
        .map(arr => (arr(0), arr(1)))  // Convert to tuples: [("frames", "5"), ...]
        .toMap
    }.getOrElse(Map.empty)

  private val asMap: Map[String, String] = parseSpecString(specString)

  import AnimationSpecification.*

  val frames: Option[Int] = asMap.get(Frames).flatMap(_.toIntOption)

  lazy val animationParameters: Map[String, StartEnd] =
    val parametersOnly = asMap -- AnimationSpecification.TIMESCALE_PARAMETERS
    val parsed = parametersOnly.map { case (k, v) => k -> parseStartEnd(v) }
    logger.debug(s"Parsed animation parameters: $parsed")
    parsed

  def isTimeSpecValid: Boolean =
    val valid = frames.exists(_ > 0)
    logger.debug(s"Time spec valid: $valid (frames=$frames)")
    valid

  def valid(spongeType: String): Boolean =
    val timeValid = isTimeSpecValid
    val paramsValid = animationParametersValid(spongeType)
    val result = timeValid && paramsValid
    logger.debug(s"Animation valid for '$spongeType': $result (timeValid=$timeValid, paramsValid=$paramsValid)")
    result

  override def toString: String =
    // Safe .get: frames guaranteed Some by AnimationSpecificationSequence validation (isTimeSpecValid check)
    val timeSpec = s"frames=${frames.get}"
    val animationSpec = animationParameters.mkString(":")
    s"$timeSpec:$animationSpec"

  private def current(bounds: StartEnd, frame: Int): Float =
    // Safe .get: frames guaranteed Some by AnimationSpecificationSequence validation (isTimeSpecValid check)
    require(frame <= frames.get, s"Frame $frame exceeds total frames ${frames.get}")
    bounds.start + (bounds.end - bounds.start) * frame / frames.get

  def level(frame: Int): Option[Float] =
    animationParameters.get(Level).map(current(_, frame))

  private def getBounds(param: String, default: Float = 0f): StartEnd =
    animationParameters.get(param).getOrElse(default, default)

  def rotationProjectionParameters(frame: Int): RotationProjectionParameters =
    require(animationParameters.nonEmpty, "AnimationSpecification.animationParameters not defined")
    RotationProjectionParameters(
      rotXW = current(getBounds(RotXW), frame),
      rotYW = current(getBounds(RotYW), frame),
      rotZW = current(getBounds(RotZW), frame),
      screenW = current(getBounds(ProjectionScreenW, Const.defaultScreenW), frame),
      eyeW = current(getBounds(ProjectionEyeW, Const.defaultEyeW), frame),
      rotX = current(getBounds(RotX), frame),
      rotY = current(getBounds(RotY), frame),
      rotZ = current(getBounds(RotZ), frame)
    )

  private def isAxisAnimated(value: Float, paramName: String): Boolean =
    value != 0 && animationParameters.contains(paramName)

  def isRotationAxisSet(x: Float, y: Float, z: Float, xw: Float, yw: Float, zw: Float): Boolean =
    isAxisAnimated(x, RotX) || isAxisAnimated(y, RotY) || isAxisAnimated(z, RotZ) ||
    isAxisAnimated(xw, RotXW) || isAxisAnimated(yw, RotYW) || isAxisAnimated(zw, RotZW)

  def hasRotationAxisConflict(x: Float, y: Float, z: Float, xw: Float, yw: Float, zw: Float): Boolean =
    isRotationAxisSet(x, y, z, xw, yw, zw)
    
  /**
   * Parses a range specification string into start and end values.
   * 
   * Format: "start-end" where start and end are floating point numbers
   * 
   * Example: "0-90" parses to (0.0f, 90.0f)
   *          "1.5-3.7" parses to (1.5f, 3.7f)
   * 
   * @param valueString Range specification in "start-end" format
   * @return Tuple of (start, end) float values
   * @throws IllegalArgumentException if format is invalid or values are not floats
   */
  private def parseStartEnd(valueString: String): StartEnd =
    val parts = valueString.split('-')
    require(parts.length == 2, s"Invalid start-end specification: $valueString")
    require(parts.head.toFloatOption.isDefined, s"Start ${parts.head} not a Float")
    require(parts.last.toFloatOption.isDefined, s"End ${parts.last} not a Float")
    (parts.head.toFloat, parts.last.toFloat)

  private def animationParametersValid(spongeType: String): Boolean =
    val validParams = AnimationSpecification.validParameters(spongeType)
    val providedParams = animationParameters.keySet
    val hasParams = animationParameters.nonEmpty
    val allValid = providedParams.subsetOf(validParams)
    if !allValid then
      val invalidParams = providedParams -- validParams
      logger.debug(s"Invalid animation parameters for '$spongeType': $invalidParams. Valid: $validParams")
    hasParams && allValid

object AnimationSpecification:
  final val RotX = "rot-x"
  final val RotY = "rot-y"
  final val RotZ = "rot-z"
  final val RotXW = "rot-x-w"
  final val RotYW = "rot-y-w"
  final val RotZW = "rot-z-w"
  final val ProjectionScreenW = "projection-screen-w"
  final val ProjectionEyeW = "projection-eye-w"
  final val Level = "level"
  final val Frames = "frames"

  final val TIMESCALE_PARAMETERS = Set(Frames)
  final val ALWAYS_VALID_PARAMETERS = Set(RotX, RotY, RotZ)
  final val FOUR_D_VALID_PARAMETERS = Set(RotXW, RotYW, RotZW, ProjectionScreenW, ProjectionEyeW)
  final val FRACTAL_VALID_PARAMETERS = Set(Level)
  final val FOUR_D_OBJECTS = Set("tesseract", "tesseract-sponge", "tesseract-sponge-2")
  final val FRACTAL_OBJECTS = Set("tesseract-sponge", "tesseract-sponge-2", "square", "cube")

  def validParameters(spongeType: String): Set[String] =
    ALWAYS_VALID_PARAMETERS ++ (
      if FOUR_D_OBJECTS.contains(spongeType) then FOUR_D_VALID_PARAMETERS else Set.empty) ++ (
      if FRACTAL_OBJECTS.contains(spongeType) then FRACTAL_VALID_PARAMETERS else Set.empty
    )


