package menger

import scala.util.Try

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.common.Const
import menger.optix.CausticsConfig
import menger.optix.RenderConfig
import org.rogach.scallop._
import org.rogach.scallop.exceptions._

class MengerCLIOptions(arguments: Seq[String]) extends ScallopConf(arguments) with LazyLogging:
  version("menger v0.4.0 (c) 2023-25, lene.preuss@gmail.com")
  banner("""Usage: menger [options]
           |
           |Menger sponge fractal renderer with OptiX GPU ray tracing support.
           |Run with --help for full options list.
           |""".stripMargin)

  // Custom error handling to show usage hint on errors
  override def onError(e: Throwable): Unit = e match
    case Help("") =>
      builder.printHelp()
      sys.exit(0)
    case Version =>
      builder.vers.foreach(println)
      sys.exit(0)
    case Exit() =>
      sys.exit(0)
    case ScallopException(message) =>
      // Print error with usage hint
      Console.err.println(s"Error: $message")
      Console.err.println()
      Console.err.println("Usage: menger [options]")
      Console.err.println("Run with --help for full options list.")
      sys.exit(1)
    case other =>
      Console.err.println(s"Error: ${other.getMessage}")
      Console.err.println()
      Console.err.println("Usage: menger [options]")
      Console.err.println("Run with --help for full options list.")
      sys.exit(1)

  // Option groups for organized help output
  private val generalGroup = group("General:")
  private val spongeGroup = group("Sponge Rendering:")
  private val projectionGroup = group("4D Projection:")
  private val animationGroup = group("Animation:")
  private val optixGroup = group("OptiX Renderer:")
  private val optixCameraGroup = group("OptiX Camera:")
  private val optixLightingGroup = group("OptiX Lighting:")
  private val optixSceneGroup = group("OptiX Scene:")
  private val optixQualityGroup = group("OptiX Quality:")
  private val optixCausticsGroup = group("OptiX Caustics:")

  private def validateSpongeType(spongeType: String): Boolean =
    isValidSpongeType(spongeType)

  // Note: "sphere" removed - use --object sphere for OptiX rendering
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


  // === General Options ===
  val timeout: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0), group = generalGroup,
    descr = "Run for N seconds then exit (0 = interactive)"
  )
  val width: ScallopOption[Int] = opt[Int](
    required = false, default = Some(Const.defaultWindowWidth), group = generalGroup,
    descr = "Window width in pixels"
  )
  val height: ScallopOption[Int] = opt[Int](
    required = false, default = Some(Const.defaultWindowHeight), group = generalGroup,
    descr = "Window height in pixels"
  )
  val saveName: ScallopOption[String] = opt[String](
    required = false, validate = _.nonEmpty, group = generalGroup,
    descr = "Save rendered image to file"
  )
  val logLevel: ScallopOption[String] = opt[String](
    required = false, default = Some("INFO"), group = generalGroup,
    validate = level => Set("ERROR", "WARN", "INFO", "DEBUG", "TRACE").contains(level.toUpperCase),
    descr = "Log level: ERROR, WARN, INFO, DEBUG, TRACE"
  )
  val profileMinMs: ScallopOption[Int] = opt[Int](
    required = false, validate = _ >= 0, group = generalGroup,
    descr = "Log frames taking longer than N ms"
  )
  val fpsLogInterval: ScallopOption[Int] = opt[Int](
    required = false, default = Some(Const.fpsLogIntervalMs), validate = _ > 0, group = generalGroup,
    descr = "FPS logging interval in ms"
  )
  val stats: ScallopOption[Boolean] = opt[Boolean](
    required = false, default = Some(false), group = generalGroup,
    descr = "Show ray tracing statistics"
  )

  // === Sponge Rendering Options ===
  val spongeType: ScallopOption[String] = opt[String](
    required = false, default = Some("square"), group = spongeGroup,
    validate = validateSpongeType,
    descr = "Sponge type: square, cube, tesseract-sponge, tesseract-sponge-2, composite[...]"
  )
  val level: ScallopOption[Float] = opt[Float](
    required = false, default = Some(1.0f), validate = _ >= 0, group = spongeGroup,
    descr = "Fractal recursion level (supports fractional values)"
  )
  val lines: ScallopOption[Boolean] = opt[Boolean](
    required = false, default = Some(false), group = spongeGroup,
    descr = "Render wireframe instead of faces"
  )
  val color: ScallopOption[Color] = opt[Color](
    required = false, default = Some(Color.LIGHT_GRAY), group = spongeGroup,
    descr = "Sponge color (hex RRGGBB or R,G,B)"
  )(using colorConverter)
  val faceColor: ScallopOption[Color] = opt[Color](
    required = false, group = spongeGroup,
    descr = "Face color (requires --line-color)"
  )(using colorConverter)
  val lineColor: ScallopOption[Color] = opt[Color](
    required = false, group = spongeGroup,
    descr = "Line color (requires --face-color)"
  )(using colorConverter)
  val antialiasSamples: ScallopOption[Int] = opt[Int](
    required = false, default = Some(Const.defaultAntialiasSamples), group = spongeGroup,
    descr = "OpenGL antialiasing samples"
  )

  // === 4D Projection Options ===
  val projectionScreenW: ScallopOption[Float] = opt[Float](
    required = false, default = Some(Const.defaultScreenW), validate = _ > 0, group = projectionGroup,
    descr = "4D projection screen W coordinate"
  )
  val projectionEyeW: ScallopOption[Float] = opt[Float](
    required = false, default = Some(Const.defaultEyeW), validate = _ > 0, group = projectionGroup,
    descr = "4D projection eye W coordinate"
  )
  val rotX: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0), validate = a => a >= 0 && a < 360, group = projectionGroup,
    descr = "Rotation around X axis (degrees)"
  )
  val rotY: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0), validate = a => a >= 0 && a < 360, group = projectionGroup,
    descr = "Rotation around Y axis (degrees)"
  )
  val rotZ: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0), validate = a => a >= 0 && a < 360, group = projectionGroup,
    descr = "Rotation around Z axis (degrees)"
  )
  val rotXW: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0), validate = a => a >= 0 && a < 360, group = projectionGroup,
    descr = "Rotation in X-W plane (degrees)"
  )
  val rotYW: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0), validate = a => a >= 0 && a < 360, group = projectionGroup,
    descr = "Rotation in Y-W plane (degrees)"
  )
  val rotZW: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0), validate = a => a >= 0 && a < 360, group = projectionGroup,
    descr = "Rotation in Z-W plane (degrees)"
  )

  // === Animation Options ===
  val animate: ScallopOption[AnimationSpecifications] = opt[AnimationSpecifications](
    group = animationGroup,
    descr = "Animation spec: frames=N:param=start-end[:param2=...] (mutually exclusive with --timeout)"
  )(using animationSpecificationsConverter)

  // === OptiX Renderer Options ===
  val optix: ScallopOption[Boolean] = opt[Boolean](
    required = false, default = Some(false), group = optixGroup,
    descr = "Use OptiX GPU ray tracing (requires --object)"
  )
  // Legacy single object option (deprecated - use objects instead)
  val objectType: ScallopOption[String] = opt[String](
    name = "object", required = false, default = None, group = optixGroup,
    validate = obj => Set("sphere", "cube", "sponge-volume", "sponge-surface").contains(obj.toLowerCase),
    descr = "Single object (legacy): sphere, cube, sponge-volume, sponge-surface"
  )

  // New multi-object option with keyword=value format
  val objects: ScallopOption[List[ObjectSpec]] = opt[List[ObjectSpec]](
    name = "objects", required = false, group = optixGroup,
    descr = "Objects (repeatable): type=TYPE:pos=x,y,z:size=S[:level=L][:color=#RGB][:ior=I]"
  )(using objectSpecConverter)
  val radius: ScallopOption[Float] = opt[Float](
    required = false, default = Some(1.0f), validate = _ > 0, group = optixGroup,
    descr = "Object radius"
  )
  val ior: ScallopOption[Float] = opt[Float](
    required = false, default = Some(1.0f), validate = _ > 0, group = optixGroup,
    descr = "Index of refraction (1.0 = opaque, 1.5 = glass)"
  )
  val scale: ScallopOption[Float] = opt[Float](
    required = false, default = Some(1.0f), validate = _ > 0, group = optixGroup,
    descr = "Object scale factor"
  )
  val center: ScallopOption[Vector3] = opt[Vector3](
    required = false, default = Some(Vector3(0f, 0f, 0f)), group = optixGroup,
    descr = "Object center position (x,y,z)"
  )(using vector3Converter)

  // === OptiX Camera Options ===
  val cameraPos: ScallopOption[Vector3] = opt[Vector3](
    required = false, default = Some(Vector3(0f, 0.5f, 3.0f)), group = optixCameraGroup,
    descr = "Camera position (x,y,z)"
  )(using vector3Converter)
  val cameraLookat: ScallopOption[Vector3] = opt[Vector3](
    required = false, default = Some(Vector3(0f, 0f, 0f)), group = optixCameraGroup,
    descr = "Camera look-at target (x,y,z)"
  )(using vector3Converter)
  val cameraUp: ScallopOption[Vector3] = opt[Vector3](
    required = false, default = Some(Vector3(0f, 1f, 0f)), group = optixCameraGroup,
    descr = "Camera up vector (x,y,z)"
  )(using vector3Converter)

  // === OptiX Lighting Options ===
  val light: ScallopOption[List[LightSpec]] = opt[List[LightSpec]](
    required = false, group = optixLightingGroup,
    descr = "Light source (repeatable, max 8): <type>:x,y,z[:intensity[:color]]"
  )(using lightSpecConverter)
  val shadows: ScallopOption[Boolean] = opt[Boolean](
    required = false, default = Some(false), group = optixLightingGroup,
    descr = "Enable shadow rays for realistic shadows"
  )

  // === OptiX Scene Options ===
  val plane: ScallopOption[PlaneSpec] = opt[PlaneSpec](
    required = false, default = Some(PlaneSpec(Axis.Y, positive = true, -2.0f)), group = optixSceneGroup,
    descr = "Ground plane: [+-]x|y|z:value (e.g., y:-2)"
  )(using planeSpecConverter)
  val planeColor: ScallopOption[PlaneColorSpec] = opt[PlaneColorSpec](
    required = false, group = optixSceneGroup,
    descr = "Plane color: RRGGBB or RRGGBB:RRGGBB for checkered"
  )(using planeColorSpecConverter)
  val maxInstances: ScallopOption[Int] = opt[Int](
    required = false, default = Some(64), group = optixSceneGroup,
    validate = n => n > 0 && n <= 1024,
    descr = "Maximum object instances in scene (1-1024, default: 64)"
  )

  // === OptiX Quality Options ===
  val antialiasing: ScallopOption[Boolean] = opt[Boolean](
    required = false, default = Some(false), group = optixQualityGroup,
    descr = "Enable recursive adaptive antialiasing"
  )
  val aaMaxDepth: ScallopOption[Int] = opt[Int](
    required = false, default = Some(2), group = optixQualityGroup,
    validate = d => d >= 1 && d <= 4,
    descr = "Maximum AA recursion depth (1-4, default: 2)"
  )
  val aaThreshold: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0.1f), group = optixQualityGroup,
    validate = t => t >= 0.0f && t <= 1.0f,
    descr = "AA edge detection threshold (0.0-1.0, default: 0.1)"
  )

  // === OptiX Caustics Options ===
  val caustics: ScallopOption[Boolean] = opt[Boolean](
    required = false, default = Some(false), group = optixCausticsGroup,
    descr = "Enable Progressive Photon Mapping caustics"
  )
  val causticsPhotons: ScallopOption[Int] = opt[Int](
    required = false, default = Some(100000), group = optixCausticsGroup,
    validate = p => p > 0 && p <= 10000000,
    descr = "Photons per PPM iteration (default: 100000)"
  )
  val causticsIterations: ScallopOption[Int] = opt[Int](
    required = false, default = Some(10), group = optixCausticsGroup,
    validate = i => i > 0 && i <= 1000,
    descr = "Number of PPM iterations (default: 10)"
  )
  val causticsRadius: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0.1f), group = optixCausticsGroup,
    validate = r => r > 0.0f && r <= 10.0f,
    descr = "Initial photon gather radius (default: 0.1)"
  )
  val causticsAlpha: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0.7f), group = optixCausticsGroup,
    validate = a => a > 0.0f && a < 1.0f,
    descr = "PPM radius reduction factor (default: 0.7)"
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
  // OptiX requires --object or --objects option to specify geometry
  validateOpt(optix, objectType, objects) { (ox, obj, objs) =>
    val isOptiXEnabled = ox.getOrElse(false)
    val hasObjectType = obj.isDefined
    val hasObjects = objs.isDefined

    if isOptiXEnabled && !hasObjectType && !hasObjects then
      Left("--optix flag requires --object or --objects option")
    else if (hasObjectType || hasObjects) && !isOptiXEnabled then
      Left("--object/--objects option requires --optix flag")
    else if hasObjectType && hasObjects then
      Left("Cannot use both --object and --objects (use --objects only)")
    else Right(())
  }

  validateOpt(shadows, optix) { (sh, ox) =>
    requiresOptixFlag("shadows", sh.getOrElse(false), ox.getOrElse(false))
  }

  validateOpt(light, optix) { (l, ox) =>
    requiresOptixOption("light", l, ox.getOrElse(false)).flatMap { _ =>
      if l.isDefined && l.get.length > 8 then Left("Maximum 8 lights allowed (MAX_LIGHTS=8)")
      else Right(())
    }
  }

  validateOpt(antialiasing, optix) { (aa, ox) =>
    requiresOptixFlag("antialiasing", aa.getOrElse(false), ox.getOrElse(false))
  }

  validateOpt(aaMaxDepth, antialiasing) { (_, aa) =>
    requiresParentFlag("aa-max-depth", aaMaxDepth.isSupplied, "antialiasing", aa.getOrElse(false))
  }

  validateOpt(aaThreshold, antialiasing) { (_, aa) =>
    requiresParentFlag("aa-threshold", aaThreshold.isSupplied, "antialiasing", aa.getOrElse(false))
  }

  validateOpt(planeColor, optix) { (pc, ox) =>
    requiresOptixOption("plane-color", pc, ox.getOrElse(false))
  }

  validateOpt(maxInstances, optix) { (_, ox) =>
    requiresParentFlag("max-instances", maxInstances.isSupplied, "optix", ox.getOrElse(false))
  }

  validateOpt(caustics, optix) { (c, ox) =>
    requiresOptixFlag("caustics", c.getOrElse(false), ox.getOrElse(false))
  }

  // Note: objectType validation is handled in the combined validateOpt(optix, objectType) above

  validateOpt(causticsPhotons, caustics) { (_, c) =>
    requiresParentFlag("caustics-photons", causticsPhotons.isSupplied, "caustics", c.getOrElse(false))
  }

  validateOpt(causticsIterations, caustics) { (_, c) =>
    requiresParentFlag("caustics-iterations", causticsIterations.isSupplied, "caustics", c.getOrElse(false))
  }

  validateOpt(causticsRadius, caustics) { (_, c) =>
    requiresParentFlag("caustics-radius", causticsRadius.isSupplied, "caustics", c.getOrElse(false))
  }

  validateOpt(causticsAlpha, caustics) { (_, c) =>
    requiresParentFlag("caustics-alpha", causticsAlpha.isSupplied, "caustics", c.getOrElse(false))
  }

  verify()

  // Config accessors - create typed config objects from CLI options
  def renderConfig: RenderConfig = RenderConfig(
    shadows = shadows(),
    antialiasing = antialiasing(),
    aaMaxDepth = aaMaxDepth(),
    aaThreshold = aaThreshold()
  )

  def causticsConfig: CausticsConfig = CausticsConfig(
    enabled = caustics(),
    photonsPerIteration = causticsPhotons(),
    iterations = causticsIterations(),
    initialRadius = causticsRadius(),
    alpha = causticsAlpha()
  )

  private def validateAnimationSpecification(spec: AnimationSpecifications, spongeType: String) =
    if spec.valid(spongeType) && spec.timeSpecValid then Right(())
    else Left("Invalid animation specification")

  /** Helper for validating that a boolean flag requires --optix */
  private def requiresOptixFlag(flagName: String, flagValue: Boolean, optixEnabled: Boolean): Either[String, Unit] =
    if flagValue && !optixEnabled then Left(s"--$flagName flag requires --optix flag")
    else Right(())

  /** Helper for validating that an optional value requires --optix */
  private def requiresOptixOption[T](optionName: String, optionValue: Option[T], optixEnabled: Boolean): Either[String, Unit] =
    if optionValue.isDefined && !optixEnabled then Left(s"--$optionName flag requires --optix flag")
    else Right(())

  /** Helper for validating that a supplied option requires a parent flag */
  private def requiresParentFlag(optionName: String, isSupplied: Boolean, parentName: String, parentEnabled: Boolean): Either[String, Unit] =
    if isSupplied && !parentEnabled then Left(s"--$optionName requires --$parentName flag")
    else Right(())


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

val objectSpecConverter = new ValueConverter[List[ObjectSpec]] {
  val argType: ArgType.V = org.rogach.scallop.ArgType.LIST
  def parse(s: List[(String, List[String])]): Either[String, Option[List[ObjectSpec]]] =
    val specStrings = s.flatMap(_._2)
    if specStrings.isEmpty then Right(None)
    else
      ObjectSpec.parseAll(specStrings) match
        case Right(objects) => Right(Some(objects))
        case Left(error) => Left(error)
}
