package menger.cli

import com.typesafe.scalalogging.Logger
import menger.AnimationSpecificationSequence
import menger.ObjectSpec
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
  protected def optix: ScallopOption[Boolean]
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

  protected def registerValidationRules(): Unit =
    validationLogger.debug("Registering CLI validation rules")
    registerProjectionValidations()
    registerAnimationValidations()
    registerColorValidations()
    registerOptiXValidations()
    registerAntialiasingValidations()
    registerCausticsValidations()
    registerHeadlessValidations()

  private def registerProjectionValidations(): Unit =
    mutuallyExclusive(timeout, animate)
    validate(projectionScreenW, projectionEyeW) { (screen, eye) =>
      if eye > screen then Right(())
      else Left("eyeW must be greater than screenW")
    }
    validateOpt(fourDRotation) { rot =>
      rot.map(parseFourDRotationValues).getOrElse(Right((0f, 0f, 0f))).map(_ => ())
    }
    validateOpt(fourDRotation, rotXW, rotYW, rotZW) { (rot4D, _, _, _) =>
      if rot4D.isDefined && (rotXW.isSupplied || rotYW.isSupplied || rotZW.isSupplied) then
        Left("--rotation-4d cannot be combined with --rot-x-w, --rot-y-w, or --rot-z-w")
      else Right(())
    }

  private def registerAnimationValidations(): Unit =
    validateOpt(animate, spongeType) {
      case (Some(spec), Some(sponge)) => validateAnimationSpecification(spec, sponge)
      case _ => Right(())
    }

    validateOpt(animate, rotX, rotY, rotZ, rotXW, rotYW, rotZW) { (spec, x, y, z, xw, yw, zw) =>
      if spec.isEmpty then Right(())
      else
        val (effectiveXW, effectiveYW, effectiveZW) =
          if fourDRotation.isSupplied then
            parseFourDRotationValues(fourDRotation()).getOrElse((0f, 0f, 0f))
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

    validateOpt(optix, scene, objects) { (ox, sc, objs) =>
      val isOptiXEnabled = ox.getOrElse(false)
      val hasScene = sc.isDefined
      val hasObjects = objs.isDefined
      validationLogger.debug(
        s"OptiX validation: enabled=$isOptiXEnabled, hasScene=$hasScene, hasObjects=$hasObjects"
      )

      if isOptiXEnabled && !hasScene && !hasObjects then
        Left("--optix flag requires either --scene or --objects option. " +
          "Add --scene <scene-name> or --objects \"type=sphere:pos=0,0,0:size=1\"")
      else if (hasScene || hasObjects) && !isOptiXEnabled then
        Left("--scene/--objects option requires --optix flag. Add --optix to enable OptiX rendering")
      else Right(())
    }

    validateOpt(shadows, optix) { (sh, _) =>
      requiresOptix("shadows", sh.getOrElse(false))
    }

    validateOpt(light, optix) { (l, _) =>
      requiresOptix("light", l).flatMap { _ =>
        if l.exists(_.length > Const.maxLights) then
          Left(s"Maximum ${Const.maxLights} lights allowed (MAX_LIGHTS=${Const.maxLights}). " +
            s"You specified ${l.get.length} lights. Reduce the number of --light options")
        else Right(())
      }
    }

    validateOpt(planeColor, optix) { (pc, _) =>
      requiresOptix("plane-color", pc)
    }

    validateOpt(maxInstances, optix) { (_, ox) =>
      requiresParent("max-instances", maxInstances.isSupplied, "optix", ox.getOrElse(false))
    }

  private def registerAntialiasingValidations(): Unit =
    validateOpt(antialiasing, optix) { (aa, _) =>
      requiresOptix("antialiasing", aa.getOrElse(false))
    }

    validateOpt(aaMaxDepth, antialiasing) { (_, aa) =>
      requiresParent("aa-max-depth", aaMaxDepth.isSupplied, "antialiasing", aa.getOrElse(false))
    }

    validateOpt(aaThreshold, antialiasing) { (_, aa) =>
      requiresParent("aa-threshold", aaThreshold.isSupplied, "antialiasing", aa.getOrElse(false))
    }

  private def registerCausticsValidations(): Unit =
    validateOpt(caustics, optix) { (c, _) =>
      requiresOptix("caustics", c.getOrElse(false))
    }

    validateOpt(causticsPhotons, caustics) { (_, c) =>
      requiresParent("caustics-photons", causticsPhotons.isSupplied, "caustics", c.getOrElse(false))
    }

    validateOpt(causticsIterations, caustics) { (_, c) =>
      requiresParent(
        "caustics-iterations", causticsIterations.isSupplied, "caustics", c.getOrElse(false)
      )
    }

    validateOpt(causticsRadius, caustics) { (_, c) =>
      requiresParent("caustics-radius", causticsRadius.isSupplied, "caustics", c.getOrElse(false))
    }

    validateOpt(causticsAlpha, caustics) { (_, c) =>
      requiresParent("caustics-alpha", causticsAlpha.isSupplied, "caustics", c.getOrElse(false))
    }

  private def hasConflictingColorOptions: Boolean =
    color.isSupplied && (faceColor.isSupplied || lineColor.isSupplied)

  private def hasFaceLineColorMismatch: Boolean =
    faceColor.isSupplied != lineColor.isSupplied

  private def hasLinesWithColorConflict: Boolean =
    lines.isSupplied && (faceColor.isSupplied || lineColor.isSupplied)

  private def validateAnimationSpecification(
    spec: AnimationSpecificationSequence, spongeType: String
  ): Either[String, Unit] =
    validationLogger.debug(s"Validating animation spec for spongeType='$spongeType': $spec")
    val isValid = spec.valid(spongeType)
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

  private def requiresOptix(flagName: String, flagValue: Boolean): Either[String, Unit] =
    requires(flagName, flagValue, "optix", optix())

  private def requiresOptix[T](flagName: String, optionValue: Option[T]): Either[String, Unit] =
    requires(flagName, optionValue.isDefined, "optix", optix())

  private def requiresParent(
    optionName: String,
    isSupplied: Boolean,
    parentName: String,
    parentEnabled: Boolean
  ): Either[String, Unit] =
    requires(optionName, isSupplied, parentName, parentEnabled)

  protected def parseFourDRotationValues(s: String): Either[String, (Float, Float, Float)] =
    val parts = s.split(",").map(_.trim)
    if parts.length != 3 then
      Left(s"--rotation-4d must be XW,YW,ZW (three comma-separated degrees, e.g., 30,20,0), got: '$s'")
    else
      val results = parts.map { p =>
        try Right(p.toFloat)
        catch case _: NumberFormatException => Left(p)
      }
      results.find(_.isLeft) match
        case Some(Left(bad)) => Left(s"--rotation-4d: '$bad' is not a valid number")
        case _ =>
          val floats = results.collect { case Right(f) => f }
          if floats.exists(f => f < 0 || f >= 360) then
            Left("--rotation-4d values must each be in range [0, 360)")
          else Right((floats(0), floats(1), floats(2)))

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
