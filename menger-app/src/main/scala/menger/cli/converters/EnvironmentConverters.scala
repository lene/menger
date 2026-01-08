package menger.cli.converters

import scala.util.Try

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.math.Vector3
import menger.ColorConversions
import menger.cli.Axis
import menger.cli.LightSpec
import menger.cli.LightType
import menger.cli.PlaneColorSpec
import menger.cli.PlaneSpec
import menger.cli.converters.ConverterUtils.unwrapTryEither
import org.rogach.scallop.ArgType
import org.rogach.scallop.ValueConverter

given planeSpecConverter: ValueConverter[PlaneSpec] with
  val argType: ArgType.V = ArgType.SINGLE

  def parse(s: List[(String, List[String])]): Either[String, Option[PlaneSpec]] =
    if s.isEmpty || s.head._2.isEmpty then Right(None)
    else
      val input = s.head._2.head.trim
      unwrapTryEither(Try(parsePlaneSpec(input)).recover {
        case e: Exception =>
          Left(s"Plane spec '$input' not recognized: ${e.getMessage}. " +
            "Expected format: [+-]?x|y|z:<value> (e.g., y:-2 or -z:5.5)")
      })

  private def parsePlaneSpec(input: String): Either[String, Option[PlaneSpec]] =
    // Plane spec pattern: [+-]?<axis>:<value>
    // - ([+-]?): optional sign indicating normal direction (group 1)
    // - ([xyz]): axis perpendicular to plane (group 2)
    // - (-?\d+\.?\d*): position value on axis, can be negative (group 3)
    // Example: "y:-2" (plane at y=-2, normal in +Y direction)
    //          "-z:5.5" (plane at z=5.5, normal in -Z direction)
    val pattern = """([+-]?)([xyz]):(-?\d+\.?\d*)""".r
    input match
      case pattern(sign, axisStr, valueStr) =>
        val axis = axisStr.toLowerCase match
          case "x" => Axis.X
          case "y" => Axis.Y
          case "z" => Axis.Z
        val positive = sign != "-"
        val value = valueStr.toFloat
        Right(Some(PlaneSpec(axis, positive, value)))
      case _ =>
        Left(
          s"Plane spec '$input' must match format [+-]?x|y|z:[-]<value> " +
          "(e.g., y:-2, +y:-2, or -z:5.5)"
        )

given planeColorSpecConverter: ValueConverter[PlaneColorSpec] with
  val argType: ArgType.V = ArgType.SINGLE

  def parse(s: List[(String, List[String])]): Either[String, Option[PlaneColorSpec]] =
    if s.isEmpty || s.head._2.isEmpty then Right(None)
    else
      val input = s.head._2.head.trim
      unwrapTryEither(Try(parseSpec(input)).recover {
        case e: Exception =>
          Left(s"Plane color '$input' not recognized: ${e.getMessage}. " +
            "Expected hex color RRGGBB or checkered pattern RRGGBB:RRGGBB")
      })

  private def parseSpec(input: String): Either[String, Option[PlaneColorSpec]] =
    if input.contains(':') then parseCheckered(input)
    else parseSolid(input)

  private def parseSolid(input: String): Either[String, Option[PlaneColorSpec]] =
    parseHexColor(input.stripPrefix("#")).map(c => Some(PlaneColorSpec(c, None)))

  private def parseCheckered(input: String): Either[String, Option[PlaneColorSpec]] =
    val parts = input.split(":")
    if parts.length != 2 then
      Left(
        "Checkered plane color must have exactly two colors separated by ':' (e.g., RRGGBB:RRGGBB)"
      )
    else
      for
        c1 <- parseHexColor(parts(0).stripPrefix("#"))
        c2 <- parseHexColor(parts(1).stripPrefix("#"))
      yield Some(PlaneColorSpec(c1, Some(c2)))

  private def parseHexColor(hex: String): Either[String, menger.common.Color] =
    if hex.length != 6 then
      Left(s"Color '$hex' must be exactly 6 hex digits (RRGGBB). Example: FF0000 for red")
    else
      Try(menger.common.Color.fromHex(hex)).toEither.left
        .map(_ => s"Color '$hex' contains invalid hex digits. Use only 0-9 and A-F")

given lightSpecConverter: ValueConverter[List[LightSpec]] with
  val argType: ArgType.V = ArgType.LIST

  def parse(s: List[(String, List[String])]): Either[String, Option[List[LightSpec]]] =
    val specStrings = s.flatMap(_._2)
    if specStrings.isEmpty then Right(None)
    else parseLightSpecs(specStrings)

  private def parseLightSpecs(
    specStrings: List[String]
  ): Either[String, Option[List[LightSpec]]] =
    // Light spec pattern: <type>:x,y,z[:intensity[:color]]
    // - (?i): case-insensitive matching
    // - (directional|point): light type (captured group 1)
    // - (-?\d+\.?\d*): x,y,z coordinates, can be negative floats (groups 2-4)
    // - (?::([^:]*)): optional intensity (group 5)
    // - (?::([^:]+)): optional color as hex or RGB (group 6)
    // Example: "directional:0,1,-1:2.0:ffffff" or "point:0,5,0"
    val pattern =
      """(?i)(directional|point):(-?\d+\.?\d*),(-?\d+\.?\d*),(-?\d+\.?\d*)(?::([^:]*))?(?::([^:]+))?""".r
    specStrings.map { input =>
      input.trim match
        case pattern(typeStr, x, y, z, intensityStr, colorStr) =>
          parseSingleLightSpec(input, typeStr, x, y, z, intensityStr, colorStr)
        case _ =>
          Left(
            s"Light spec '$input' has invalid format. " +
            "Expected: <type>:x,y,z[:intensity[:color]] where type is 'directional' or 'point'. " +
            "Examples: directional:0,1,-1 or point:0,5,0:2.0:ffffff"
          )
    }.foldLeft[Either[String, List[LightSpec]]](Right(List.empty)) {
      case (Right(acc), Right(spec)) => Right(acc :+ spec)
      case (Left(err), _) => Left(err)
      case (_, Left(err)) => Left(err)
    }.map(Some(_))

  private def parseSingleLightSpec(
    input: String,
    typeStr: String,
    x: String,
    y: String,
    z: String,
    intensityStr: String,
    colorStr: String
  ): Either[String, LightSpec] =
    Try {
      val lightType = typeStr.toLowerCase match
        case "directional" => LightType.DIRECTIONAL
        case "point" => LightType.POINT
      val position = Vector3(x.toFloat, y.toFloat, z.toFloat)
      val intensity = Option(intensityStr).filter(_.nonEmpty).map(_.toFloat).getOrElse(1.0f)
      val color = Option(colorStr).map(parseColor).getOrElse(Color.WHITE)
      LightSpec(lightType, position, intensity, color)
    }.toEither.left.map { e =>
      s"Light spec '$input' parse error: ${e.getMessage}. " +
        "Check that coordinates and intensity are valid numbers"
    }

  private def parseColor(colorStr: String): Color =
    if colorStr.contains(',') then
      val parts = colorStr.split(",").map(_.trim.toInt)
      ColorConversions.rgbIntsToColor(parts)
    else Color.valueOf(colorStr)
