package menger.common

/**
 * Base exception for all Menger application errors.
 *
 * Provides a consistent exception hierarchy for better error handling and debugging.
 * All custom exceptions extend this base class.
 *
 * @param message Human-readable error description
 * @param cause Optional underlying cause
 */
sealed abstract class MengerException(
  message: String,
  cause: Option[Throwable] = None
) extends RuntimeException(message, cause.orNull)

/**
 * Exception thrown when parsing input fails due to invalid format.
 *
 * Use this for syntax errors in user input (CLI arguments, config files, etc.)
 *
 * @param message Description of what went wrong
 * @param input The input string that failed to parse
 * @param expected Optional description of expected format
 */
case class ParseException(
  message: String,
  input: String,
  expected: Option[String] = None
) extends MengerException(
  expected match
    case Some(exp) => s"$message. Input: '$input'. Expected: $exp"
    case None => s"$message. Input: '$input'"
)

/**
 * Exception thrown when input has valid syntax but invalid values.
 *
 * Use this for semantic errors (e.g., negative radius, unknown enum value).
 *
 * @param message Description of the validation failure
 * @param field Name of the field that failed validation
 * @param value The invalid value
 * @param constraint Optional description of the constraint that was violated
 */
case class ValidationException(
  message: String,
  field: String,
  value: Any,
  constraint: Option[String] = None
) extends MengerException(
  constraint match
    case Some(c) => s"$message. Field: '$field', Value: '$value'. Constraint: $c"
    case None => s"$message. Field: '$field', Value: '$value'"
)

/**
 * Exception thrown when configuration is invalid or incomplete.
 *
 * Use this for errors in application configuration.
 *
 * @param message Description of the configuration error
 * @param configKey Optional configuration key that caused the error
 */
case class ConfigurationException(
  message: String,
  configKey: Option[String] = None
) extends MengerException(
  configKey match
    case Some(key) => s"$message. Config key: '$key'"
    case None => message
)

/**
 * Exception thrown when a geometry type is not recognized.
 *
 * @param geometryType The unrecognized geometry type string
 * @param validTypes List of valid geometry types for the error message
 */
case class UnknownGeometryException(
  geometryType: String,
  validTypes: Seq[String] = Seq.empty
) extends MengerException(
  if validTypes.nonEmpty then
    s"Unknown geometry type: '$geometryType'. Valid types: ${validTypes.mkString(", ")}"
  else
    s"Unknown geometry type: '$geometryType'"
)

/**
 * Exception thrown when animation specification is invalid.
 *
 * @param message Description of the animation error
 * @param specIndex Optional index of the problematic specification in a sequence
 */
case class AnimationException(
  message: String,
  specIndex: Option[Int] = None
) extends MengerException(
  specIndex match
    case Some(idx) => s"$message (specification index: $idx)"
    case None => message
)

/**
 * Exception thrown when a direction vector is invalid.
 *
 * @param x X component
 * @param y Y component
 * @param z Z component
 * @param reason Why the direction is invalid
 */
case class InvalidDirectionException(
  x: Float,
  y: Float,
  z: Float,
  reason: String = "invalid direction vector"
) extends MengerException(s"Invalid direction ($x, $y, $z): $reason")
