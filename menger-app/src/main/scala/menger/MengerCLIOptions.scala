package menger

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.cli.Axis
import menger.cli.CliValidation
import menger.cli.LightSpec
import menger.cli.PlaneColorSpec
import menger.cli.PlaneSpec
import menger.cli.converters.animationSpecificationSequenceConverter
import menger.cli.converters.colorConverter
import menger.cli.converters.lightSpecConverter
import menger.cli.converters.objectSpecConverter
import menger.cli.converters.planeColorSpecConverter
import menger.cli.converters.planeSpecConverter
import menger.cli.converters.vector3Converter
import menger.common.Const
import menger.common.ObjectType
import menger.optix.CausticsConfig
import menger.optix.RenderConfig
import org.rogach.scallop._
import org.rogach.scallop.exceptions._

class MengerCLIOptions(arguments: Seq[String])
    extends ScallopConf(arguments)
    with CliValidation
    with LazyLogging:

  version("menger v0.5.0 (c) 2023-26, lene.preuss@gmail.com")
  banner("""Usage: menger [options]
           |
           |Menger sponge fractal renderer with OptiX GPU ray tracing support.
           |Run with --help for full options list.
           |""".stripMargin)

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

  // Sponge type validation
  private val basicSpongeTypes = List(
    "cube", "square", "square-sponge", "cube-sponge",
    "tesseract", "tesseract-sponge", "tesseract-sponge-2"
  )

  private def isValidSpongeType(spongeType: String): Boolean =
    if basicSpongeTypes.contains(spongeType) then true
    else spongeType match
      case common.Patterns.CompositeType(content) =>
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
  val headless: ScallopOption[Boolean] = opt[Boolean](
    name = "headless", default = Some(false), group = generalGroup,
    descr = "Render without displaying window (requires --save-name)"
  )

  // === Sponge Rendering Options ===
  val spongeType: ScallopOption[String] = opt[String](
    required = false, default = Some("square"), group = spongeGroup,
    validate = isValidSpongeType,
    descr = "Sponge type: square, cube, tesseract-sponge-volume, tesseract-sponge-surface, composite[...] (old names: tesseract-sponge, tesseract-sponge-2 still work)"
  )
  val level: ScallopOption[Float] = opt[Float](
    required = false, default = Some(Const.defaultSpongeLevel), validate = _ >= 0, group = spongeGroup,
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
  val animate: ScallopOption[AnimationSpecificationSequence] = opt[AnimationSpecificationSequence](
    group = animationGroup,
    descr = "Animation spec: frames=N:param=start-end[:param2=...] (mutually exclusive with --timeout)"
  )(using animationSpecificationSequenceConverter)

  // === OptiX Renderer Options ===
  val optix: ScallopOption[Boolean] = opt[Boolean](
    required = false, default = Some(false), group = optixGroup,
    descr = "Use OptiX GPU ray tracing (requires --objects)"
  )

  val scene: ScallopOption[String] = opt[String](
    name = "scene", required = false, group = optixGroup,
    descr = "Load pre-compiled DSL scene by name (e.g., 'glass-sphere') or fully-qualified class name (e.g., 'examples.dsl.GlassSphere'). " +
      "Mutually exclusive with --objects"
  )

  val objects: ScallopOption[List[ObjectSpec]] = opt[List[ObjectSpec]](
    name = "objects", required = false, group = optixGroup,
    descr = "Objects (repeatable): type=TYPE[:param=value...]. " +
      s"Types: ${ObjectType.validTypesString}. " +
      "Common: pos=x,y,z, size=S, color=#RGB, ior=I, material=PRESET, texture=FILE, emission=E. " +
      "3D sponge: level=L. " +
      "4D sponge: level=L (required). " +
      "4D projection: rot-xw, rot-yw, rot-zw, eye-w, screen-w. " +
      "4D edges: edge-radius=R, edge-material=PRESET, edge-color=#RGB, edge-emission=E"
  )(using objectSpecConverter)

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
    descr = "Light source (repeatable, max 8): <type>:x,y,z[:intensity[:color]]. For directional: x,y,z points TO light"
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
    required = false, default = Some(Const.defaultMaxInstances), group = optixSceneGroup,
    validate = n => n > 0 && n <= Const.maxInstancesLimit,
    descr = s"Maximum object instances in scene (1-${Const.maxInstancesLimit}, " +
      s"default: auto-calculated for edge rendering, otherwise ${Const.defaultMaxInstances})"
  )

  // Track if user explicitly provided maxInstances value
  def userSetMaxInstances: Boolean = maxInstances.isSupplied

  val textureDir: ScallopOption[String] = opt[String](
    required = false, default = Some("."), group = optixSceneGroup,
    descr = "Base directory for texture files (default: current directory)"
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
    required = false, default = Some(Const.maxPhotonsDefault), group = optixCausticsGroup,
    validate = p => p > 0 && p <= Const.maxPhotonsLimit,
    descr = s"Photons per PPM iteration (default: ${Const.maxPhotonsDefault})"
  )
  val causticsIterations: ScallopOption[Int] = opt[Int](
    required = false, default = Some(Const.maxIterationsDefault), group = optixCausticsGroup,
    validate = i => i > 0 && i <= Const.maxIterationsLimit,
    descr = s"Number of PPM iterations (default: ${Const.maxIterationsDefault})"
  )
  val causticsRadius: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0.1f), group = optixCausticsGroup,
    validate = r => r > 0.0f && r <= Const.maxCausticsRadius,
    descr = "Initial photon gather radius (default: 0.1)"
  )
  val causticsAlpha: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0.7f), group = optixCausticsGroup,
    validate = a => a > 0.0f && a < 1.0f,
    descr = "PPM radius reduction factor (default: 0.7)"
  )

  // Register validation rules from CliValidation trait
  registerValidationRules()
  verify()

  // Config accessors
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
