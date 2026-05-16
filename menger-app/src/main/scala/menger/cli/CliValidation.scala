package menger.cli

import com.typesafe.scalalogging.Logger
import menger.AnimationSpecificationSequence
import menger.ObjectSpec
import menger.cli.converters.ConverterUtils
import menger.common.Const
import org.rogach.scallop.ScallopConf
import org.rogach.scallop.ScallopOption

trait CliValidation:
  self: ScallopConf =>

  private val validationLogger = Logger(getClass)

  protected def timeout: ScallopOption[Float]
  protected def animate: ScallopOption[AnimationSpecificationSequence]
  protected def spongeType: ScallopOption[String]
  protected def level: ScallopOption[Float]
  protected def rotX: ScallopOption[Float]
  protected def rotY: ScallopOption[Float]
  protected def rotZ: ScallopOption[Float]
  protected def rotXW: ScallopOption[Float]
  protected def rotYW: ScallopOption[Float]
  protected def rotZW: ScallopOption[Float]
  protected def fourDRotation: ScallopOption[String]
  protected def projectionScreenW: ScallopOption[Float]
  protected def projectionEyeW: ScallopOption[Float]
  protected def color: ScallopOption[?]
  protected def faceColor: ScallopOption[?]
  protected def lineColor: ScallopOption[?]
  protected def lines: ScallopOption[Boolean]
  protected def scene: ScallopOption[String]
  protected def objects: ScallopOption[List[ObjectSpec]]
  protected def shadows: ScallopOption[Boolean]
  protected def light: ScallopOption[List[LightSpec]]
  protected def antialiasing: ScallopOption[Boolean]
  protected def aaMaxDepth: ScallopOption[Int]
  protected def aaThreshold: ScallopOption[Float]
  protected def planeColor: ScallopOption[PlaneColorSpec]
  protected def maxInstances: ScallopOption[Int]
  protected def caustics: ScallopOption[Boolean]
  protected def causticsPhotons: ScallopOption[Int]
  protected def causticsIterations: ScallopOption[Int]
  protected def causticsRadius: ScallopOption[Float]
  protected def causticsAlpha: ScallopOption[Float]
  protected def headless: ScallopOption[Boolean]
  protected def saveName: ScallopOption[String]
  protected def freezeT: ScallopOption[Float]
  protected def startT: ScallopOption[Float]
  protected def endT: ScallopOption[Float]
  protected def tFrames: ScallopOption[Int]
  protected def video: ScallopOption[String]
  protected def videoQuality: ScallopOption[Int]
  protected def keepFrames: ScallopOption[Boolean]

  protected def registerValidationRules(): Unit =
    validationLogger.debug("Registering CLI validation rules")
    registerProjectionValidations()
    registerAnimationValidations()
    registerColorValidations()
    registerOptiXValidations()
    registerAntialiasingValidations()
    registerCausticsValidations()
    registerHeadlessValidations()
    registerTAnimationValidations()
    registerVideoValidations()

  private def registerProjectionValidations(): Unit =
    mutuallyExclusive(timeout, animate)
    validate(projectionScreenW, projectionEyeW) { (screen, eye) =>
      if eye > screen then Right(())
      else Left("eyeW must be greater than screenW")
    }
    validateOpt(fourDRotation) { rot =>
      rot.map(ConverterUtils.parseFourDRotation).getOrElse(Right((0f, 0f, 0f))).map(_ => ())
    }
    validateOpt(fourDRotation) { rot4D =>
      if rot4D.isDefined && (rotXW.isSupplied || rotYW.isSupplied || rotZW.isSupplied) then
        Left("--rotation-4d cannot be combined with --rot-x-w, --rot-y-w, or --rot-z-w")
      else Right(())
    }

  private def registerAnimationValidations(): Unit =
    validateOpt(animate, spongeType, objects) { (specOpt, spongeOpt, objsOpt) =>
      specOpt match
        case Some(spec) =>
          val types: Set[String] = objsOpt match
            case Some(objs) if objs.nonEmpty => objs.map(_.objectType).toSet
            case _ => spongeOpt.toSet
          validateAnimationSpecification(spec, types)
        case None => Right(())
    }

    validateOpt(animate, rotX, rotY, rotZ, rotXW, rotYW, rotZW) { (spec, x, y, z, xw, yw, zw) =>
      if spec.isEmpty then Right(())
      else
        val (effectiveXW, effectiveYW, effectiveZW) =
          if fourDRotation.isSupplied then
            ConverterUtils.parseFourDRotation(fourDRotation()).getOrElse((0f, 0f, 0f))
          else
            (xw.getOrElse(0f), yw.getOrElse(0f), zw.getOrElse(0f))
        if spec.get.hasRotationAxisConflict(
          x.getOrElse(0), y.getOrElse(0), z.getOrElse(0),
          effectiveXW, effectiveYW, effectiveZW
        ) then Left("Animation specification has rotation axis set that is also set statically")
        else Right(())
    }

    validateOpt(animate, level) { (spec, lvl) =>
      if spec.isEmpty then Right(())
      else
        val levelIsAnimated = spec.get.parts.exists(_.animationParameters.contains("level"))
        if levelIsAnimated && level.isSupplied then
          Left("Level cannot be specified both as --level option and in animation specification")
        else Right(())
    }

  private def registerColorValidations(): Unit =
    validateOpt(color, faceColor, lineColor) { (_, _, _) =>
      if hasConflictingColorOptions then
        Left("--color cannot be used together with --face-color or --line-color. " +
          "Use either --color OR (--face-color AND --line-color)")
      else if hasFaceLineColorMismatch then
        Left("--face-color and --line-color must be specified together. " +
          "Provide both options or use --color instead")
      else Right(())
    }

    validateOpt(lines, faceColor, lineColor) { (_, _, _) =>
      if hasLinesWithColorConflict then
        Left("--lines cannot be used together with --face-color or --line-color")
      else Right(())
    }

  private def registerOptiXValidations(): Unit =
    // Validate scene and objects are mutually exclusive
    mutuallyExclusive(scene, objects)

    validateOpt(light) { l =>
      if l.exists(_.length > Const.maxLights) then
        Left(s"Maximum ${Const.maxLights} lights allowed (MAX_LIGHTS=${Const.maxLights}). " +
          s"You specified ${l.get.length} lights. Reduce the number of --light options")
      else Right(())
    }

  private def registerAntialiasingValidations(): Unit =
    validateOpt(aaMaxDepth, antialiasing) { (_, aa) =>
      requires("aa-max-depth", aaMaxDepth.isSupplied, "antialiasing", aa.getOrElse(false))
    }

    validateOpt(aaThreshold, antialiasing) { (_, aa) =>
      requires("aa-threshold", aaThreshold.isSupplied, "antialiasing", aa.getOrElse(false))
    }

  private def requiresCausticsFlag[A](optName: String, opt: ScallopOption[A]): Unit =
    validateOpt(opt, caustics) { (_, c) =>
      requires(optName, opt.isSupplied, "caustics", c.getOrElse(false))
    }

  private def registerCausticsValidations(): Unit =
    requiresCausticsFlag("caustics-photons", causticsPhotons)
    requiresCausticsFlag("caustics-iterations", causticsIterations)
    requiresCausticsFlag("caustics-radius", causticsRadius)
    requiresCausticsFlag("caustics-alpha", causticsAlpha)

  private def hasConflictingColorOptions: Boolean =
    color.isSupplied && (faceColor.isSupplied || lineColor.isSupplied)

  private def hasFaceLineColorMismatch: Boolean =
    faceColor.isSupplied != lineColor.isSupplied

  private def hasLinesWithColorConflict: Boolean =
    lines.isSupplied && (faceColor.isSupplied || lineColor.isSupplied)

  private def validateAnimationSpecification(
    spec: AnimationSpecificationSequence, spongeTypes: Set[String]
  ): Either[String, Unit] =
    validationLogger.debug(s"Validating animation spec for spongeTypes=$spongeTypes: $spec")
    val isValid = spec.valid(spongeTypes)
    val timeValid = spec.isTimeSpecValid
    validationLogger.debug(s"Animation validation result: isValid=$isValid, timeValid=$timeValid")
    if isValid && timeValid then Right(())
    else Left(
      "Invalid animation specification. Check that: " +
      "(1) animation parameters are valid for the object type, " +
      "(2) time format is correct (frames=N or fps=N), " +
      "(3) parameter values are within valid ranges"
    )

  private def requires(
    optionName: String,
    isSupplied: Boolean,
    parentName: String,
    parentEnabled: Boolean
  ): Either[String, Unit] =
    if isSupplied && !parentEnabled then Left(s"--$optionName requires --$parentName flag")
    else Right(())

  private def registerHeadlessValidations(): Unit =
    validateOpt(headless, saveName) { (h, s) =>
      if h.getOrElse(false) && s.isEmpty then
        Left("--headless requires --save-name to specify output file")
      else Right(())
    }

    validateOpt(headless, timeout) { (h, t) =>
      if h.getOrElse(false) && t.getOrElse(0f) > 0f then
        Left("--headless and --timeout are mutually exclusive")
      else Right(())
    }

  private def registerTAnimationValidations(): Unit =
    // --t is mutually exclusive with --start-t, --end-t, --frames
    validateOpt(freezeT) { _ =>
      if freezeT.isSupplied && (startT.isSupplied || endT.isSupplied || tFrames.isSupplied) then
        Left("--t is mutually exclusive with --start-t, --end-t, and --frames")
      else Right(())
    }

    // --t requires --scene
    validateOpt(freezeT, scene) { (ft, sc) =>
      if ft.isDefined && sc.isEmpty then
        Left("--t requires --scene (animated DSL scene)")
      else Right(())
    }

    // --frames requires --scene
    validateOpt(tFrames, scene) { (fr, sc) =>
      if fr.isDefined && sc.isEmpty then
        Left("--frames requires --scene (animated DSL scene)")
      else Right(())
    }

    // --t and --frames are mutually exclusive with --animate
    validateOpt(freezeT, animate) { (ft, an) =>
      if ft.isDefined && an.isDefined then
        Left("--t is mutually exclusive with --animate")
      else Right(())
    }

    validateOpt(tFrames, animate) { (fr, an) =>
      if fr.isDefined && an.isDefined then
        Left("--frames is mutually exclusive with --animate")
      else Right(())
    }

    // --frames requires --save-name containing %
    validateOpt(tFrames, saveName) { (fr, sn) =>
      if fr.isDefined then
        sn match
          case Some(name) if name.contains("%") => Right(())
          case _ => Left("--frames requires --save-name containing '%' for frame numbering (e.g., frame_%04d.png)")
      else Right(())
    }

  private def registerVideoValidations(): Unit =
    // --video-quality and --keep-frames require --video
    validateOpt(videoQuality, video) { (_, v) =>
      requires("video-quality", videoQuality.isSupplied, "video", v.isDefined)
    }
    validateOpt(keepFrames, video) { (_, v) =>
      requires("keep-frames", keepFrames.isSupplied, "video", v.isDefined)
    }

    // --video requires --frames
    validateOpt(video, tFrames) { (v, fr) =>
      if v.isDefined && fr.isEmpty then
        Left("--video requires --frames (animation must be specified)")
      else Right(())
    }

    // --video requires --save-name with %
    validateOpt(video, saveName) { (v, sn) =>
      if v.isDefined then
        sn match
          case Some(name) if name.contains("%") => Right(())
          case _ => Left("--video requires --save-name containing '%' for frame numbering (e.g., frame_%04d.png)")
      else Right(())
    }

    // --video extension must be .mp4 or .mkv
    validateOpt(video) { v =>
      v match
        case Some(path) if path.toLowerCase.endsWith(".mp4") || path.toLowerCase.endsWith(".mkv") =>
          Right(())
        case Some(path) =>
          Left(s"--video '$path': unsupported format. Use .mp4 (H.264) or .mkv (HEVC/hevc_nvenc)")
        case None => Right(())
    }

    // --video requires ffmpeg and the encoder for the chosen format
    validateOpt(video) { v =>
      v match
        case None => Right(())
        case Some(outputPath) =>
          try
            menger.engines.VideoEncoder.checkAvailable(outputPath)
            Right(())
          catch case e: IllegalStateException => Left(e.getMessage)
    }
