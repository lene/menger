package menger.cli

import menger.AnimationSpecificationSequence
import menger.ObjectSpec
import menger.common.Const
import org.rogach.scallop.ScallopConf
import org.rogach.scallop.ScallopOption

trait CliValidation:
  self: ScallopConf =>

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
  protected def objectType: ScallopOption[String]
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

  protected def registerValidationRules(): Unit =
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

    validateOpt(optix, objectType, objects) { (ox, obj, objs) =>
      val isOptiXEnabled = ox.getOrElse(false)
      val hasObjectType = obj.isDefined
      val hasObjects = objs.isDefined

      if isOptiXWithoutObjects(isOptiXEnabled, hasObjectType, hasObjects) then
        Left("--optix flag requires --object or --objects option. " +
          "Add --object sphere or --objects \"type=sphere:pos=0,0,0\"")
      else if hasObjectsWithoutOptiX(isOptiXEnabled, hasObjectType, hasObjects) then
        Left("--object/--objects option requires --optix flag. Add --optix to enable OptiX rendering")
      else if hasBothObjectOptions(hasObjectType, hasObjects) then
        Left("Cannot use both --object and --objects (use --objects only). " +
          "Combine multiple objects with --objects \"obj1:obj2:obj3\"")
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

    validateOpt(antialiasing, optix) { (aa, _) =>
      requiresOptix("antialiasing", aa.getOrElse(false))
    }

    validateOpt(aaMaxDepth, antialiasing) { (_, aa) =>
      requiresParent("aa-max-depth", aaMaxDepth.isSupplied, "antialiasing", aa.getOrElse(false))
    }

    validateOpt(aaThreshold, antialiasing) { (_, aa) =>
      requiresParent("aa-threshold", aaThreshold.isSupplied, "antialiasing", aa.getOrElse(false))
    }

    validateOpt(planeColor, optix) { (pc, _) =>
      requiresOptix("plane-color", pc)
    }

    validateOpt(maxInstances, optix) { (_, ox) =>
      requiresParent("max-instances", maxInstances.isSupplied, "optix", ox.getOrElse(false))
    }

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

  private def isOptiXWithoutObjects(
    isOptiXEnabled: Boolean, hasObjectType: Boolean, hasObjects: Boolean
  ): Boolean =
    isOptiXEnabled && !hasObjectType && !hasObjects

  private def hasObjectsWithoutOptiX(
    isOptiXEnabled: Boolean, hasObjectType: Boolean, hasObjects: Boolean
  ): Boolean =
    (hasObjectType || hasObjects) && !isOptiXEnabled

  private def hasBothObjectOptions(hasObjectType: Boolean, hasObjects: Boolean): Boolean =
    hasObjectType && hasObjects

  private def validateAnimationSpecification(
    spec: AnimationSpecificationSequence, spongeType: String
  ): Either[String, Unit] =
    if spec.valid(spongeType) && spec.isTimeSpecValid then Right(())
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
