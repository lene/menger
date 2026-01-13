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
  protected def projectionScreenW: ScallopOption[Float]
  protected def projectionEyeW: ScallopOption[Float]
  protected def color: ScallopOption[?]
  protected def faceColor: ScallopOption[?]
  protected def lineColor: ScallopOption[?]
  protected def lines: ScallopOption[Boolean]
  protected def optix: ScallopOption[Boolean]
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

  private def registerAnimationValidations(): Unit =
    validateOpt(animate, spongeType) {
      case (Some(spec), Some(sponge)) => validateAnimationSpecification(spec, sponge)
      case _ => Right(())
    }

    validateOpt(animate, rotX, rotY, rotZ, rotXW, rotYW, rotZW) { (spec, x, y, z, xw, yw, zw) =>
      if spec.isEmpty then Right(())
      else if spec.get.hasRotationAxisConflict(
        x.getOrElse(0), y.getOrElse(0), z.getOrElse(0),
        xw.getOrElse(0), yw.getOrElse(0), zw.getOrElse(0)
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
    validateOpt(optix, objects) { (ox, objs) =>
      val isOptiXEnabled = ox.getOrElse(false)
      val hasObjects = objs.isDefined
      validationLogger.debug(
        s"OptiX validation: enabled=$isOptiXEnabled, hasObjects=$hasObjects"
      )

      if isOptiXEnabled && !hasObjects then
        Left("--optix flag requires --objects option. " +
          "Add --objects \"type=sphere:pos=0,0,0:size=1\"")
      else if hasObjects && !isOptiXEnabled then
        Left("--objects option requires --optix flag. Add --optix to enable OptiX rendering")
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
