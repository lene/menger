package menger

import scala.jdk.CollectionConverters._
import scala.jdk.OptionConverters._

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.cli.CliValidation
import menger.cli.FogSpec
import menger.cli.LightSpec
import menger.cli.PlaneColorSpec
import menger.cli.PlaneSpec
import menger.cli.converters.ConverterUtils
import menger.cli.converters.animationSpecificationSequenceConverter
import menger.cli.converters.colorConverter
import menger.cli.converters.fogSpecConverter
import menger.cli.converters.lightSpecConverter
import menger.cli.converters.objectSpecConverter
import menger.cli.converters.planeColorSpecConverter
import menger.cli.converters.planeSpecConverter
import menger.cli.converters.vector3Converter
import menger.common.CausticsConfig
import menger.common.Const
import menger.common.ObjectType
import menger.common.RenderConfig
import menger.common.RenderLimits
import menger.config.CrossConfig
import menger.dsl.DenoiseMode
import org.rogach.scallop._
import org.rogach.scallop.exceptions._

/** Thrown by [[MengerCLIOptions.onError]] instead of calling sys.exit directly.
 *  [[Main]] catches this and exits with the given code.
 */
final class MengerExitException(val code: Int, message: String)
    extends RuntimeException(message)

class MengerCLIOptions(arguments: Seq[String])
    extends ScallopConf(arguments)
    with CliValidation
    with LazyLogging:

  version("menger v0.8.1 (c) 2023-26, lene.preuss@gmail.com")
  banner("""Usage: menger [options]
           |
           |Menger sponge fractal renderer with OptiX GPU ray tracing support.
           |Run with --help for full options list.
           |""".stripMargin)

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  override def onError(e: Throwable): Unit = e match
    case Help("") =>
      builder.printHelp()
      throw MengerExitException(0, "help requested")
    case Version =>
      builder.vers.foreach(v => logger.info(v))
      throw MengerExitException(0, "version requested")
    case Exit() =>
      throw MengerExitException(0, "exit requested")
    case ScallopException(message) =>
      logger.error(s"Error: $message")
      logger.error("Usage: menger [options]")
      logger.error("Run with --help for full options list.")
      throw MengerExitException(1, s"CLI error: $message")
    case other =>
      logger.error(s"Error: ${other.getMessage}")
      logger.error("Usage: menger [options]")
      logger.error("Run with --help for full options list.")
      throw MengerExitException(1, s"Unexpected CLI error: ${other.getMessage}")

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
  val stats: ScallopOption[Boolean] = toggle(
    default = Some(false), group = generalGroup,
    descrYes = "Show ray tracing statistics"
  )
  val statsJson: ScallopOption[String] = opt[String](
    name = "stats-json", required = false, group = generalGroup,
    descr = "Write last-frame render stats as JSON to this file (implies --stats)"
  )
  val headless: ScallopOption[Boolean] = opt[Boolean](
    name = "headless", default = Some(false), group = generalGroup,
    descr = "Render without displaying window (requires --save-name)"
  )

  // === Coordinate Cross ===
  val cross: ScallopOption[Boolean] = opt[Boolean](
    required = false, default = Some(false), group = generalGroup,
    descr = "Show coordinate cross (XYZ axis visualization)"
  )
  val crossLength: ScallopOption[Double] = opt[Double](
    required = false, default = Some(2.0), group = generalGroup,
    descr = "Coordinate cross half-length from origin (default: 2.0)"
  )
  val crossThickness: ScallopOption[Double] = opt[Double](
    required = false, default = Some(0.03), group = generalGroup,
    descr = "Coordinate cross cylinder radius (default: 0.03)"
  )
  val crossMaterial: ScallopOption[String] = opt[String](
    required = false, group = generalGroup,
    descr = "Material preset for coordinate cross (chrome, metal, gold, etc.)"
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

  private val isDegree: Float => Boolean = a => a >= 0 && a < 360

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
    required = false, default = Some(0), validate = isDegree, group = projectionGroup,
    descr = "Rotation around X axis (degrees)"
  )
  val rotY: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0), validate = isDegree, group = projectionGroup,
    descr = "Rotation around Y axis (degrees)"
  )
  val rotZ: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0), validate = isDegree, group = projectionGroup,
    descr = "Rotation around Z axis (degrees)"
  )
  val rotXW: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0), validate = isDegree, group = projectionGroup,
    descr = "Rotation in X-W plane (degrees)"
  )
  val rotYW: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0), validate = isDegree, group = projectionGroup,
    descr = "Rotation in Y-W plane (degrees)"
  )
  val rotZW: ScallopOption[Float] = opt[Float](
    required = false, default = Some(0), validate = isDegree, group = projectionGroup,
    descr = "Rotation in Z-W plane (degrees)"
  )
  val fourDRotation: ScallopOption[String] = opt[String](
    name = "rotation-4d", required = false, group = projectionGroup,
    descr = "4D rotation shorthand: XW,YW,ZW in degrees (e.g., --rotation-4d=30,20,0). " +
      "Mutually exclusive with --rot-xw, --rot-yw, --rot-zw"
  )

  // Resolve effective 4D rotation angles, honouring --rotation-4d shorthand.
  // Validation has already run by the time these accessors are called, so the
  // parse result is guaranteed to be Right if fourDRotation.isSupplied.
  private lazy val parsedFourDRotation: Option[(Float, Float, Float)] =
    fourDRotation.toOption.map(ConverterUtils.parseFourDRotation(_).getOrElse((0f, 0f, 0f)))

  def effectiveRotXW: Float = parsedFourDRotation.map(_._1).getOrElse(rotXW())
  def effectiveRotYW: Float = parsedFourDRotation.map(_._2).getOrElse(rotYW())
  def effectiveRotZW: Float = parsedFourDRotation.map(_._3).getOrElse(rotZW())

  // === Animation Options ===
  val animate: ScallopOption[AnimationSpecificationSequence] = opt[AnimationSpecificationSequence](
    group = animationGroup,
    descr = "Animation spec: frames=N:param=start-end[:param2=...] (mutually exclusive with --timeout)"
  )(using animationSpecificationSequenceConverter)

  // === Scene Animation (t-parameter) ===
  private val tAnimationGroup = group("Scene Animation (t-parameter):")

  val freezeT: ScallopOption[Float] = opt[Float](
    name = "t", required = false, group = tAnimationGroup,
    descr = "Evaluate animated scene at fixed t value (freeze-frame). Requires --scene"
  )
  val startT: ScallopOption[Float] = opt[Float](
    name = "start-t", required = false, default = Some(0f), group = tAnimationGroup,
    descr = "Start value for t-parameter animation range (default: 0)"
  )
  val endT: ScallopOption[Float] = opt[Float](
    name = "end-t", required = false, default = Some(1f), group = tAnimationGroup,
    descr = "End value for t-parameter animation range (default: 1)"
  )
  val tFrames: ScallopOption[Int] = opt[Int](
    name = "frames", required = false, validate = _ > 0, group = tAnimationGroup,
    descr = "Number of frames in t-parameter animation (requires --scene, --save-name with %)"
  )
  val preview: ScallopOption[Boolean] = opt[Boolean](
    name = "preview", required = false, default = Some(false), group = tAnimationGroup,
    descr = "Interactive animation preview: scrub t with Left/Right, Shift+Left/Right for larger steps, Space to play/pause, Home/End to jump. Requires --scene with an animated scene"
  )

  // === Video Output Options ===
  private val videoGroup = group("Video Output:")

  val video: ScallopOption[String] = opt[String](
    name = "video", required = false, group = videoGroup,
    descr = "Encode frame sequence into video file. Supported formats: .mp4 (H.264/libx264), .mkv (HEVC/hevc_nvenc). Requires --frames and --save-name"
  )
  val videoQuality: ScallopOption[Int] = opt[Int](
    name = "video-quality", required = false, default = Some(12), group = videoGroup,
    validate = q => q >= 0 && q <= 51,
    descr = "Video QP quality value (0=lossless, 51=worst; default 12 = master quality)"
  )
  val keepFrames: ScallopOption[Boolean] = opt[Boolean](
    name = "keep-frames", required = false, default = Some(false), group = videoGroup,
    descr = "Keep individual frame PNG files after video encoding (default: delete frames)"
  )

  // === OptiX Renderer Options ===
  val scene: ScallopOption[String] = opt[String](
    name = "scene", required = false, group = optixGroup,
    descr = "Load scene by short name (e.g., 'glass-sphere'), fully-qualified class name (e.g., 'examples.dsl.GlassSphere'), " +
      "or path to a .scala file for runtime compilation (e.g., 'my_scene.scala'). Mutually exclusive with --objects"
  )

  val objects: ScallopOption[List[ObjectSpec]] = opt[List[ObjectSpec]](
    name = "objects", required = false, group = optixGroup,
    descr = "Objects (repeatable): type=TYPE[:param=value...]. " +
      s"Types: ${ObjectType.validTypesString}. " +
      "Common: pos=x,y,z, size=S, color=#RGB, ior=I, material=PRESET, texture=FILE, emission=E, film-thickness=NM. " +
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
    descr = "Light source (repeatable, max 8). Types: directional:x,y,z[:intensity[:color]] (x,y,z points TO light), point:x,y,z[:intensity[:color]], area:px,py,pz:nx,ny,nz:radius[:samples[:intensity[:color[:shape]]]] (disk emitter, soft shadows)"
  )(using lightSpecConverter)
  val shadows: ScallopOption[Boolean] = opt[Boolean](
    required = false, default = Some(true), group = optixLightingGroup,
    descr = "Enable shadow rays for realistic shadows"
  )
  val transparentShadows: ScallopOption[Boolean] = opt[Boolean](
    name = "transparent-shadows",
    required = false, default = Some(false), group = optixLightingGroup,
    descr = "Enable colored light tinting through transparent/glass objects (requires --shadows)"
  )

  // === OptiX Scene Options ===
  val plane: ScallopOption[PlaneSpec] = opt[PlaneSpec](
    required = false, group = optixSceneGroup,
    descr = "Ground plane: [+-]x|y|z:value (e.g., y:-2)"
  )(using planeSpecConverter)
  val planeColor: ScallopOption[PlaneColorSpec] = opt[PlaneColorSpec](
    required = false, group = optixSceneGroup,
    descr = "Plane color: RRGGBB or RRGGBB:RRGGBB for checkered"
  )(using planeColorSpecConverter)
  val planeMaterial: ScallopOption[String] = opt[String](
    name = "plane-material", required = false, group = optixSceneGroup,
    validate = name => menger.common.Material.fromName(name).isPresent,
    descr = s"Plane material preset name (${menger.common.Material.presetNames.asScala.mkString(", ")})"
  )
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

  val envMap: ScallopOption[String] = opt[String](
    required = false, group = optixSceneGroup,
    descr = "Path to HDR equirectangular environment map (.hdr)"
  )

  val fog: ScallopOption[FogSpec] = opt[FogSpec](
    name = "fog", required = false, group = optixSceneGroup,
    descr = "Fog/depth cue: density=0.05:color=0.8,0.8,0.9"
  )(using fogSpecConverter)

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
  val maxRayDepth: ScallopOption[Int] = opt[Int](
    required = false, default = Some(RenderLimits.MaxRayDepth), group = optixQualityGroup,
    validate = d => d >= 1 && d <= RenderLimits.MaxRayDepth,
    descr = s"Maximum ray bounce depth (1-${RenderLimits.MaxRayDepth}, default: ${RenderLimits.MaxRayDepth})"
  )
  val denoise: ScallopOption[Boolean] = toggle(
    name = "denoise",
    default = Some(false),
    group = optixQualityGroup,
    descrYes = "Denoise the final accumulated frame",
    descrNo = "Disable final-frame denoising"
  )
  val noDenoise: ScallopOption[Boolean] = opt[Boolean](
    name = "no-denoise",
    required = false,
    default = Some(false),
    group = optixQualityGroup,
    descr = "Disable final-frame denoising, overriding DSL scenes"
  )
  val accumulationFrames: ScallopOption[Int] = opt[Int](
    required = false, default = Some(1), group = optixQualityGroup,
    descr = "Temporal accumulation frame count (>=1). >1 averages N frames for noise reduction.",
    validate = _ >= 1
  )
  val allowUniformRender: ScallopOption[Boolean] = opt[Boolean](
    required = false, default = Some(false), group = optixQualityGroup,
    descr = "Allow renders where >=99% of pixels are the same colour (otherwise treated as a failure)"
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
    required = false, group = optixCausticsGroup,
    validate = r => r > 0.0f && r <= Const.maxCausticsRadius,
    descr = "Initial photon gather radius (default: auto-derived from scene geometry)"
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
    transparentShadows = transparentShadows(),
    antialiasing = antialiasing(),
    aaMaxDepth = aaMaxDepth(),
    aaThreshold = aaThreshold(),
    maxRayDepth = maxRayDepth()
  )

  def denoiseMode: DenoiseMode =
    if noDenoise() then DenoiseMode.Off
    else if denoise() then DenoiseMode.Final
    else DenoiseMode.Off

  def denoiseModeSupplied: Boolean = denoise.isSupplied || noDenoise.isSupplied

  def causticsConfig: CausticsConfig = CausticsConfig(
    enabled = caustics(),
    photonsPerIteration = causticsPhotons(),
    iterations = causticsIterations(),
    initialRadius = causticsRadius.toOption.getOrElse(CausticsConfig.AutoRadius),
    alpha = causticsAlpha()
  )

  def crossConfig: CrossConfig = CrossConfig(
    enabled = cross(),
    length = crossLength().toFloat,
    thickness = crossThickness().toFloat,
    material = crossMaterial.toOption.flatMap(s => menger.common.Material.fromName(s).toScala)
  )
