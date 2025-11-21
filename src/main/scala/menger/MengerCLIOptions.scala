package menger

import scala.util.Try

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.common.Const
import org.rogach.scallop._

class MengerCLIOptions(arguments: Seq[String]) extends ScallopConf(arguments) with LazyLogging:
  version("menger v0.3.6 (c) 2023-25, lene.preuss@gmail.com")

  private def validateSpongeType(spongeType: String): Boolean =
    isValidSpongeType(spongeType)

  private val basicSpongeTypes = List("cube", "square", "square-sponge", "cube-sponge", "tesseract", "tesseract-sponge", "tesseract-sponge-2", "sphere")
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
  val optix: ScallopOption[Boolean] = opt[Boolean](required = false, default = Some(false))
  val radius: ScallopOption[Float] = opt[Float](required = false, default = Some(1.0f), validate = _ > 0)
  val ior: ScallopOption[Float] = opt[Float](required = false, default = Some(1.0f), validate = _ > 0)
  val scale: ScallopOption[Float] = opt[Float](required = false, default = Some(1.0f), validate = _ > 0)
  val color: ScallopOption[Color] = opt[Color](required = false, default = Some(Color.LIGHT_GRAY))(
    using colorConverter
  )
  val faceColor: ScallopOption[Color] = opt[Color](required = false)(
    using colorConverter
  )
  val lineColor: ScallopOption[Color] = opt[Color](required = false)(
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
  val profileMinMs: ScallopOption[Int] = opt[Int](
    required = false, validate = _ >= 0
  )
  val fpsLogInterval: ScallopOption[Int] = opt[Int](
    required = false, default = Some(Const.fpsLogIntervalMs), validate = _ > 0
  )
  val logLevel: ScallopOption[String] = opt[String](
    required = false, default = Some("INFO"),
    validate = level => Set("ERROR", "WARN", "INFO", "DEBUG", "TRACE").contains(level.toUpperCase)
  )
  val stats: ScallopOption[Boolean] = opt[Boolean](
    required = false, default = Some(false)
  )
  val shadows: ScallopOption[Boolean] = opt[Boolean](
    required = false, default = Some(false),
    descr = "Enable shadow rays for realistic shadows (OptiX only)"
  )

  val antialiasing: ScallopOption[Boolean] = opt[Boolean](
    required = false, default = Some(false),
    descr = "Enable recursive adaptive antialiasing (OptiX only)"
  )

  val aaMaxDepth: ScallopOption[Int] = opt[Int](
    required = false, default = Some(2),
    validate = d => d >= 1 && d <= 4,
    descr = "Maximum AA recursion depth (1-4, default: 2)"
  )

  val aaThreshold: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0.1f),
    validate = t => t >= 0.0f && t <= 1.0f,
    descr = "AA edge detection threshold (0.0-1.0, default: 0.1)"
  )

  // Camera parameters
  val cameraPos: ScallopOption[Vector3] = opt[Vector3](
    required = false, default = Some(Vector3(0f, 0.5f, 3.0f))
  )(using vector3Converter)
  val cameraLookat: ScallopOption[Vector3] = opt[Vector3](
    required = false, default = Some(Vector3(0f, 0f, 0f))
  )(using vector3Converter)
  val cameraUp: ScallopOption[Vector3] = opt[Vector3](
    required = false, default = Some(Vector3(0f, 1f, 0f))
  )(using vector3Converter)

  // Scene geometry parameters
  val center: ScallopOption[Vector3] = opt[Vector3](
    required = false, default = Some(Vector3(0f, 0f, 0f))
  )(using vector3Converter)
  val plane: ScallopOption[PlaneSpec] = opt[PlaneSpec](
    required = false, default = Some(PlaneSpec(Axis.Y, positive = true, -2.0f))
  )(using planeSpecConverter)
  val planeColor: ScallopOption[PlaneColorSpec] = opt[PlaneColorSpec](
    required = false,
    descr = "Plane color: #RRGGBB for solid, or RRGGBB:RRGGBB for checkered (OptiX only)"
  )(using planeColorSpecConverter)

  // Lighting parameters (OptiX only)
  val light: ScallopOption[List[LightSpec]] = opt[List[LightSpec]](
    required = false,
    descr = "Light source (repeatable, max 8). Format: <type>:x,y,z[:intensity[:color]] where type is directional|point"
  )(using lightSpecConverter)

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

  validateOpt(animate, level) { (spec, lvl) =>
    if spec.isEmpty then Right(())
    else
      val levelIsAnimated = spec.get.parts.exists(_.animationParameters.contains("level"))
      if levelIsAnimated && level.isSupplied then
        Left("Level cannot be specified both as --level option and in animation specification")
      else Right(())
  }

  // Validate color option combinations
  validateOpt(color, faceColor, lineColor) { (c, fc, lc) =>
    if color.isSupplied && (faceColor.isSupplied || lineColor.isSupplied) then
      Left("--color cannot be used together with --face-color or --line-color")
    else if (faceColor.isSupplied && !lineColor.isSupplied) || (!faceColor.isSupplied && lineColor.isSupplied) then
      Left("--face-color and --line-color must be specified together")
    else Right(())
  }

  validateOpt(lines, faceColor, lineColor) { (l, fc, lc) =>
    if lines.isSupplied && (faceColor.isSupplied || lineColor.isSupplied) then
      Left("--lines cannot be used together with --face-color or --line-color")
    else Right(())
  }

  // Validate OptiX-related options
  validateOpt(spongeType, optix) { (st, ox) =>
    if st.contains("sphere") && !ox.getOrElse(false) then
      Left("--sponge-type sphere requires --optix flag")
    else if ox.getOrElse(false) && !st.contains("sphere") then
      Left("--optix flag requires --sponge-type sphere")
    else Right(())
  }

  validateOpt(shadows, optix) { (sh, ox) =>
    if sh.getOrElse(false) && !ox.getOrElse(false) then
      Left("--shadows flag requires --optix flag")
    else Right(())
  }

  validateOpt(light, optix) { (l, ox) =>
    if l.isDefined && !ox.getOrElse(false) then
      Left("--light flag requires --optix flag")
    else if l.isDefined && l.get.length > 8 then
      Left("Maximum 8 lights allowed (MAX_LIGHTS=8)")
    else Right(())
  }

  validateOpt(antialiasing, optix) { (aa, ox) =>
    if aa.getOrElse(false) && !ox.getOrElse(false) then
      Left("--antialiasing flag requires --optix flag")
    else Right(())
  }

  validateOpt(aaMaxDepth, antialiasing) { (_, aa) =>
    if aaMaxDepth.isSupplied && !aa.getOrElse(false) then
      Left("--aa-max-depth requires --antialiasing flag")
    else Right(())
  }

  validateOpt(aaThreshold, antialiasing) { (_, aa) =>
    if aaThreshold.isSupplied && !aa.getOrElse(false) then
      Left("--aa-threshold requires --antialiasing flag")
    else Right(())
  }

  validateOpt(planeColor, optix) { (pc, ox) =>
    if pc.isDefined && !ox.getOrElse(false) then
      Left("--plane-color flag requires --optix flag")
    else Right(())
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

val vector3Converter = new ValueConverter[Vector3] {
  val argType: ArgType.V = org.rogach.scallop.ArgType.SINGLE
  def parse(s: List[(String, List[String])]): Either[String, Option[Vector3]] =
    if s.isEmpty || s.head._2.isEmpty then Right(None)
    else
      val input = s.head._2.head.trim
      Try {
        val parts = input.split(",").map(_.trim.toFloat)
        if parts.length != 3 then
          Left(s"Vector3 '$input' must have exactly 3 components (x,y,z)")
        else
          Right(Some(Vector3(parts(0), parts(1), parts(2))))
      }.recover {
        case e: NumberFormatException => Left(s"Vector3 '$input' contains non-numeric values")
        case e: Exception => Left(s"Vector3 '$input' not recognized: ${e.getMessage}")
      }.get
}

enum Axis:
  case X, Y, Z

case class PlaneSpec(axis: Axis, positive: Boolean, value: Float)

val planeSpecConverter = new ValueConverter[PlaneSpec] {
  val argType: ArgType.V = org.rogach.scallop.ArgType.SINGLE
  def parse(s: List[(String, List[String])]): Either[String, Option[PlaneSpec]] =
    if s.isEmpty || s.head._2.isEmpty then Right(None)
    else
      val input = s.head._2.head.trim
      Try {
        val pattern = """([+-]?)([xyz]):(-?\d+\.?\d*)""".r
        input match
          case pattern(sign, axisStr, valueStr) =>
            val axis = axisStr.toLowerCase match
              case "x" => Axis.X
              case "y" => Axis.Y
              case "z" => Axis.Z
            val positive = sign != "-"  // Default to + if no sign given
            val value = valueStr.toFloat
            Right(Some(PlaneSpec(axis, positive, value)))
          case _ =>
            Left(s"Plane spec '$input' must match format [+-]?x|y|z:[-]<value> (e.g., y:-2, +y:-2, or -z:5.5)")
      }.recover {
        case e: Exception => Left(s"Plane spec '$input' not recognized: ${e.getMessage}")
      }.get
}

enum LightType:
  case DIRECTIONAL, POINT

case class LightSpec(lightType: LightType, position: Vector3, intensity: Float, color: Color)

val lightSpecConverter = new ValueConverter[List[LightSpec]] {
  val argType: ArgType.V = org.rogach.scallop.ArgType.LIST
  def parse(s: List[(String, List[String])]): Either[String, Option[List[LightSpec]]] =
    val specStrings = s.flatMap(_._2)
    if specStrings.isEmpty then Right(None)
    else
      val pattern = """(?i)(directional|point):(-?\d+\.?\d*),(-?\d+\.?\d*),(-?\d+\.?\d*)(?::([^:]*))?(?::([^:]+))?""".r
      specStrings.map { input =>
        input.trim match
          case pattern(typeStr, x, y, z, intensityStr, colorStr) =>
            Try {
              val lightType = typeStr.toLowerCase match
                case "directional" => LightType.DIRECTIONAL
                case "point" => LightType.POINT
              val position = Vector3(x.toFloat, y.toFloat, z.toFloat)
              val intensity = Option(intensityStr).filter(_.nonEmpty).map(_.toFloat).getOrElse(1.0f)
              val color = Option(colorStr).map { c =>
                if c.contains(',') then
                  val parts = c.split(",").map(_.trim.toInt)
                  val Array(r, g, b, a) = parts.map(_ / 255f).padTo(4, 1f)
                  Color(r, g, b, a)
                else
                  Color.valueOf(c)
              }.getOrElse(Color.WHITE)
              LightSpec(lightType, position, intensity, color)
            }.toEither.left.map(e => s"Light spec '$input' not recognized: ${e.getMessage}")
          case _ =>
            Left(s"Light spec '$input' must match format <type>:x,y,z[:intensity[:color]] where type is directional|point (e.g., directional:0,1,-1, point:0,5,0:2.0:ffffff)")
      }.foldLeft[Either[String, List[LightSpec]]](Right(List.empty)) {
        case (Right(acc), Right(spec)) => Right(acc :+ spec)
        case (Left(err), _) => Left(err)
        case (_, Left(err)) => Left(err)
      }.map(Some(_))
}

// Plane color specification: solid (one color) or checkered (two colors)
// Uses menger.common.Color for compatibility with OptiX renderer API
case class PlaneColorSpec(color1: menger.common.Color, color2: Option[menger.common.Color]):
  def isSolid: Boolean = color2.isEmpty
  def isCheckered: Boolean = color2.isDefined

val planeColorSpecConverter = new ValueConverter[PlaneColorSpec] {
  val argType: ArgType.V = org.rogach.scallop.ArgType.SINGLE

  def parse(s: List[(String, List[String])]): Either[String, Option[PlaneColorSpec]] =
    if s.isEmpty || s.head._2.isEmpty then Right(None)
    else
      val input = s.head._2.head.trim
      Try { parseSpec(input) }.recover {
        case e: Exception => Left(s"Plane color '$input' not recognized: ${e.getMessage}")
      }.get

  private def parseSpec(input: String): Either[String, Option[PlaneColorSpec]] =
    if input.contains(':') then parseCheckered(input)
    else parseSolid(input)

  private def parseSolid(input: String): Either[String, Option[PlaneColorSpec]] =
    parseHexColor(input.stripPrefix("#")).map(c => Some(PlaneColorSpec(c, None)))

  private def parseCheckered(input: String): Either[String, Option[PlaneColorSpec]] =
    val parts = input.split(":")
    if parts.length != 2 then
      Left("Checkered plane color must have exactly two colors separated by ':' (e.g., RRGGBB:RRGGBB)")
    else
      for
        c1 <- parseHexColor(parts(0).stripPrefix("#"))
        c2 <- parseHexColor(parts(1).stripPrefix("#"))
      yield Some(PlaneColorSpec(c1, Some(c2)))

  private def parseHexColor(hex: String): Either[String, menger.common.Color] =
    if hex.length != 6 then
      Left(s"Color '$hex' must be exactly 6 hex digits (RRGGBB)")
    else
      Try { menger.common.Color.fromHex(hex) }.toEither.left.map(_ => s"Color '$hex' contains invalid hex digits")
}
