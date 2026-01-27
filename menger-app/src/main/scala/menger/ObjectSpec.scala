package menger

import scala.util.Try

import com.typesafe.scalalogging.LazyLogging
import menger.common.Color
import menger.common.ObjectType
import menger.optix.Material

/**
 * Object specification parsed from CLI --object flag.
 * Format: type=TYPE:key=value:key=value...
 *
 * Success examples:
 *   type=sphere:pos=0,0,0:size=1.0:color=#FF0000:ior=1.5
 *   type=cube:pos=2,0,0:size=1.5:color=#0000FF
 *   type=sponge-volume:pos=-2,0,0:size=2.0:level=3:color=#00FF00
 *   type=sponge-surface:pos=0,2,0:size=2.0:level=2.5:color=#FFFF00
 *   type=cube-sponge:pos=1,0,0:size=2.0:level=2:color=#FF00FF
 *   type=sphere:pos=0,0,0:material=glass:ior=1.7
 *   type=cube:pos=0,0,0:material=metal:color=#FFD700
 *   type=cube:pos=0,0,0:texture=brick.png
 *   type=tesseract:pos=0,0,0:size=2:rot-xw=30:rot-yw=15:color=#4488FF
 *
 * Error examples (input → error message):
 *   "type=invalid" → "Invalid object type: 'invalid'. Valid types: sphere, cube, ..."
 *   "pos=1,2,3" (missing type) → "Missing required 'type' field..."
 *   "type=sphere:pos=1,2" → "Invalid position format: '1,2'. Expected format: pos=x,y,z..."
 *   "type=sphere:size=abc" → "Invalid size value 'abc': For input string: \"abc\"..."
 *   "type=sponge-volume:pos=0,0,0" → "Sponge object requires 'level' field..."
 *   "type=sphere:material=unknown" → "Unknown material preset: 'unknown'. Valid presets: ..."
 *   "type=sphere:color=notahex" → "Invalid color value 'notahex': For input string: \"notahex\"..."
 *   "type=tesseract:eye-w=1:screen-w=2" → "eye-w must be greater than screen-w..."
 */
case class ObjectSpec(
  objectType: String,
  x: Float = 0.0f,
  y: Float = 0.0f,
  z: Float = 0.0f,
  size: Float = 1.0f,
  level: Option[Float] = None,
  color: Option[Color] = None,
  ior: Float = 1.0f,
  material: Option[Material] = None,
  texture: Option[String] = None,
  projection4D: Option[Projection4DSpec] = None,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None
):
  /** Returns true if edge rendering parameters are specified */
  def hasEdgeRendering: Boolean = edgeRadius.isDefined || edgeMaterial.isDefined

/**
 * 4D projection parameters for hypercube objects (tesseract, etc.).
 * Only applicable to object types where ObjectType.isHypercube returns true.
 */
case class Projection4DSpec(
  eyeW: Float = Projection4DSpec.DefaultEyeW,
  screenW: Float = Projection4DSpec.DefaultScreenW,
  rotXW: Float = Projection4DSpec.DefaultRotXW,
  rotYW: Float = Projection4DSpec.DefaultRotYW,
  rotZW: Float = Projection4DSpec.DefaultRotZW
):
  require(eyeW > screenW, s"eyeW ($eyeW) must be greater than screenW ($screenW)")
  require(eyeW > 0 && screenW > 0, "eyeW and screenW must be positive")

object Projection4DSpec:
  val DefaultEyeW: Float = 3.0f
  val DefaultScreenW: Float = 1.5f
  val DefaultRotXW: Float = 15f
  val DefaultRotYW: Float = 10f
  val DefaultRotZW: Float = 0f

  val default: Projection4DSpec = Projection4DSpec()

object ObjectSpec extends LazyLogging:

  /**
   * Parse object specification from keyword=value format.
   * Format: type=TYPE:pos=x,y,z:size=S:level=L:color=#RRGGBB:ior=I:material=PRESET:texture=FILE
   *
   * Material keywords:
   *   material=PRESET - base material preset (glass, water, diamond, chrome, gold, copper, metal, plastic, matte, film, parchment)
   *   ior=VALUE       - override IOR (also applies without material preset)
   *   roughness=VALUE - override roughness (only with material preset)
   *   metallic=VALUE  - override metallic (only with material preset)
   *   specular=VALUE  - override specular (only with material preset)
   *   emission=VALUE  - override emission (0.0-10.0, default 0.0)
   *   color=#RRGGBB   - override color (works with or without material preset)
   *   texture=FILE    - texture filename (relative to texture directory)
   *
   * 4D projection keywords (only for tesseract type):
   *   eye-w=VALUE     - 4D eye W-coordinate (default: 3.0)
   *   screen-w=VALUE  - 4D screen W-coordinate (default: 1.5)
   *   rot-xw=DEGREES  - XW plane rotation angle (default: 15)
   *   rot-yw=DEGREES  - YW plane rotation angle (default: 10)
   *   rot-zw=DEGREES  - ZW plane rotation angle (default: 0)
   *
   * Edge rendering keywords (only for tesseract type):
   *   edge-radius=VALUE     - Radius of cylinder edges (default: 0.02)
   *   edge-material=PRESET  - Material preset for edges (film, parchment, etc.)
   *   edge-color=#RRGGBB    - Edge color override
   *   edge-emission=VALUE   - Edge emission override (0.0-10.0)
   *
   * Examples with edge rendering:
   *   type=tesseract:material=film:edge-material=film:edge-emission=3.0
   *   type=tesseract:material=glass:edge-color=#00FFFF:edge-emission=5.0:edge-radius=0.03
   */
  def parse(spec: String): Either[String, ObjectSpec] =
    logger.debug(s"Parsing object spec: $spec")
    val parts = spec.split(":").map(_.trim)
    val kvPairs = parts.flatMap { part =>
      part.split("=", 2) match
        case Array(key, value) => Some(key.trim.toLowerCase -> value.trim)
        case _ => None
    }.toMap
    logger.debug(s"Parsed key-value pairs: $kvPairs")

    val result = for
      objType <- parseObjectType(kvPairs)
      (x, y, z) <- parsePosition(kvPairs)
      size <- parseSize(kvPairs)
      level <- parseLevel(kvPairs)
      color <- parseColor(kvPairs)
      ior <- parseIOR(kvPairs)
      material <- parseMaterial(kvPairs, color, ior)
      texture <- parseTexture(kvPairs)
      projection4D <- parse4DProjection(kvPairs, objType)
      edgeParams <- parseEdgeParameters(kvPairs, objType)
      _ <- validateSpongeLevel(objType, level)
    yield ObjectSpec(objType, x, y, z, size, level, color, ior, material, texture, projection4D,
      edgeParams._1, edgeParams._2)

    result match
      case Right(obj) => logger.debug(s"Successfully parsed: $obj")
      case Left(err) => logger.debug(s"Parse failed: $err")
    result

  private def parseObjectType(kvPairs: Map[String, String]): Either[String, String] =
    kvPairs.get("type") match
      case Some(t) if ObjectType.isValid(t) => Right(t.toLowerCase)
      case Some(t) =>
        Left(s"Invalid object type: '$t'. Valid types: ${ObjectType.validTypesString}. " +
          "Example: type=sphere or type=cube")
      case None =>
        Left("Missing required 'type' field. Add type=<object-type> to specification. " +
          s"Valid types: ${ObjectType.validTypesString}")

  private def parsePosition(kvPairs: Map[String, String]): Either[String, (Float, Float, Float)] =
    kvPairs.get("pos") match
      case Some(posStr) =>
        posStr.split(",").map(_.trim) match
          case Array(xStr, yStr, zStr) =>
            Try((xStr.toFloat, yStr.toFloat, zStr.toFloat)).toEither.left.map { e =>
              s"Invalid position value in '$posStr': ${e.getMessage}. " +
                "Position components must be valid numbers (e.g., pos=1.0,2.0,3.0)"
            }
          case _ =>
            Left(s"Invalid position format: '$posStr'. Expected format: pos=x,y,z " +
              "(three comma-separated numbers). Example: pos=1.0,2.0,3.0")
      case None => Right((0.0f, 0.0f, 0.0f))

  private def parseSize(kvPairs: Map[String, String]): Either[String, Float] =
    kvPairs.get("size") match
      case Some(sizeStr) =>
        Try(sizeStr.toFloat).toEither.left.map { e =>
          s"Invalid size value '$sizeStr': ${e.getMessage}. Size must be a valid number (e.g., size=1.5)"
        }
      case None => Right(1.0f)

  private def parseLevel(kvPairs: Map[String, String]): Either[String, Option[Float]] =
    kvPairs.get("level") match
      case Some(levelStr) =>
        Try(levelStr.toFloat).toEither.left.map { e =>
          s"Invalid level value '$levelStr': ${e.getMessage}. Level must be a valid number (e.g., level=2)"
        }.map(Some(_))
      case None => Right(None)

  private def parseColor(kvPairs: Map[String, String]): Either[String, Option[menger.common.Color]] =
    kvPairs.get("color") match
      case Some(colorStr) =>
        Try {
          val hexStr = if colorStr.startsWith("#") then colorStr.substring(1) else colorStr
          menger.common.Color.fromHex(hexStr)
        }.toEither.left.map { e =>
          s"Invalid color value '$colorStr': ${e.getMessage}. " +
            "Color must be hex format (e.g., color=#FF0000 or color=FF0000)"
        }.map(Some(_))
      case None => Right(None)

  private def parseIOR(kvPairs: Map[String, String]): Either[String, Float] =
    kvPairs.get("ior") match
      case Some(iorStr) =>
        Try(iorStr.toFloat).toEither.left.map { e =>
          s"Invalid IOR value '$iorStr': ${e.getMessage}. " +
            "IOR (index of refraction) must be a valid number (e.g., ior=1.5)"
        }
      case None => Right(1.0f)

  private def parseMaterial(
    kvPairs: Map[String, String],
    color: Option[Color],
    ior: Float
  ): Either[String, Option[Material]] =
    kvPairs.get("material") match
      case Some(presetName) =>
        Material.fromName(presetName) match
          case Some(baseMaterial) =>
            parseMaterialOverrides(kvPairs, baseMaterial, color, ior)
          case None =>
            Left(s"Unknown material preset: '$presetName'. Valid presets: ${Material.presetNames.mkString(", ")}")
      case None =>
        Right(None)

  private def parseMaterialOverrides(
    kvPairs: Map[String, String],
    baseMaterial: Material,
    color: Option[Color],
    ior: Float
  ): Either[String, Option[Material]] =
    for
      roughness <- parseOptionalFloat(kvPairs, "roughness", "roughness value (0.0-1.0)")
      metallic <- parseOptionalFloat(kvPairs, "metallic", "metallic value (0.0-1.0)")
      specular <- parseOptionalFloat(kvPairs, "specular", "specular value (0.0-1.0)")
      emission <- parseOptionalFloat(kvPairs, "emission", "emission value (0.0-10.0)")
    yield Some(
      baseMaterial
        .withColorOpt(color)
        .withIorOpt(Option.when(kvPairs.contains("ior"))(ior))
        .withRoughnessOpt(roughness)
        .withMetallicOpt(metallic)
        .withSpecularOpt(specular)
        .withEmissionOpt(emission)
    )

  private def parseOptionalFloat(
    kvPairs: Map[String, String],
    key: String,
    description: String
  ): Either[String, Option[Float]] =
    kvPairs.get(key) match
      case Some(valueStr) =>
        Try(valueStr.toFloat).toEither.left.map { e =>
          s"Invalid $key value '$valueStr': ${e.getMessage}. Expected a valid $description"
        }.map(Some(_))
      case None => Right(None)

  private def parseTexture(kvPairs: Map[String, String]): Either[String, Option[String]] =
    kvPairs.get("texture") match
      case Some(filename) if filename.nonEmpty => Right(Some(filename))
      case Some(_) => Left("Texture filename cannot be empty. Provide a valid filename (e.g., texture=brick.png)")
      case None => Right(None)

  private def validateSpongeLevel(objType: String, level: Option[Float]): Either[String, Unit] =
    if ObjectType.isSponge(objType) && level.isEmpty then
      Left("Sponge object requires 'level' field. Add level=<number> to specification. " +
        s"Example: type=$objType:level=2")
    else if ObjectType.is4DSponge(objType) && level.isEmpty then
      Left("4D sponge object requires 'level' field. Add level=<number> to specification. " +
        s"Example: type=$objType:level=1")
    else if level.exists(_ < 0) then
      Left(s"Level must be non-negative, got ${level.get}")
    else
      Right(())

  private def parse4DProjection(
    kvPairs: Map[String, String],
    objType: String
  ): Either[String, Option[Projection4DSpec]] =
    if ObjectType.isProjected4D(objType) then
      for
        eyeW <- parseFloatParam(kvPairs, "eye-w", Projection4DSpec.DefaultEyeW, "4D eye W-coordinate")
        screenW <- parseFloatParam(kvPairs, "screen-w", Projection4DSpec.DefaultScreenW, "4D screen W-coordinate")
        rotXW <- parseFloatParam(kvPairs, "rot-xw", Projection4DSpec.DefaultRotXW, "XW rotation angle in degrees")
        rotYW <- parseFloatParam(kvPairs, "rot-yw", Projection4DSpec.DefaultRotYW, "YW rotation angle in degrees")
        rotZW <- parseFloatParam(kvPairs, "rot-zw", Projection4DSpec.DefaultRotZW, "ZW rotation angle in degrees")
        _ <- validate4DParams(eyeW, screenW)
      yield Some(Projection4DSpec(eyeW, screenW, rotXW, rotYW, rotZW))
    else
      Right(None)

  private def parseFloatParam(
    kvPairs: Map[String, String],
    key: String,
    default: Float,
    description: String
  ): Either[String, Float] =
    kvPairs.get(key) match
      case Some(valueStr) =>
        Try(valueStr.toFloat).toEither.left.map { e =>
          s"Invalid $key value '$valueStr': ${e.getMessage}. Expected a valid $description (e.g., $key=$default)"
        }
      case None => Right(default)

  private def validate4DParams(eyeW: Float, screenW: Float): Either[String, Unit] =
    if eyeW <= screenW then
      Left(s"eye-w ($eyeW) must be greater than screen-w ($screenW) for 4D projection")
    else if eyeW <= 0 || screenW <= 0 then
      Left("eye-w and screen-w must be positive values for 4D projection")
    else
      Right(())

  /**
   * Validate multiple object specifications.
   * Returns error if any spec is invalid, including index of failing spec.
   */
  def parseAll(specs: List[String]): Either[String, List[ObjectSpec]] =
    specs.zipWithIndex.foldLeft[Either[String, List[ObjectSpec]]](Right(List.empty)) {
      case (acc, (spec, index)) =>
        acc.flatMap { objects =>
          parse(spec).left.map { error =>
            s"Error in object specification #${index + 1}: $error"
          }.map(obj => objects :+ obj)
        }
    }

  // Edge parameter defaults
  private val DefaultEdgeRadius: Float = 0.02f

  /**
   * Parse edge rendering parameters (only applicable to hypercube types).
   * Returns (edgeRadius, edgeMaterial) tuple.
   */
  private def parseEdgeParameters(
    kvPairs: Map[String, String],
    objType: String
  ): Either[String, (Option[Float], Option[Material])] =
    // Edge parameters only applicable to hypercube types (tesseract)
    val hasEdgeParams = kvPairs.keys.exists(k =>
      k.startsWith("edge-") || k == "edgeradius" || k == "edgematerial" ||
      k == "edgecolor" || k == "edgeemission"
    )

    if hasEdgeParams && !ObjectType.isProjected4D(objType) then
      Left(s"Edge rendering parameters (edge-radius, edge-material, etc.) are only valid for 4D projected types (tesseract, tesseract-sponge, tesseract-sponge-2), not '$objType'")
    else if !hasEdgeParams then
      Right((None, None))
    else
      for
        edgeRadius <- parseEdgeRadius(kvPairs)
        edgeMaterial <- parseEdgeMaterial(kvPairs)
      yield (edgeRadius, edgeMaterial)

  private def parseEdgeRadius(kvPairs: Map[String, String]): Either[String, Option[Float]] =
    kvPairs.get("edge-radius") match
      case Some(valueStr) =>
        Try(valueStr.toFloat).toEither.left.map { e =>
          s"Invalid edge-radius value '$valueStr': ${e.getMessage}. " +
            s"Expected a valid radius (e.g., edge-radius=$DefaultEdgeRadius)"
        }.flatMap { value =>
          if value <= 0 then
            Left(s"edge-radius must be positive, got $value")
          else
            Right(Some(value))
        }
      case None =>
        // If any edge parameter is present, use default radius
        val hasAnyEdgeParam = kvPairs.keys.exists(k =>
          k == "edge-material" || k == "edge-color" || k == "edge-emission"
        )
        Right(if hasAnyEdgeParam then Some(DefaultEdgeRadius) else None)

  private def parseEdgeMaterial(kvPairs: Map[String, String]): Either[String, Option[Material]] =
    val edgeColor = parseEdgeColor(kvPairs)
    val edgeEmission = parseOptionalFloat(kvPairs, "edge-emission", "edge emission value (0.0-10.0)")

    kvPairs.get("edge-material") match
      case Some(presetName) =>
        Material.fromName(presetName) match
          case Some(baseMaterial) =>
            for
              color <- edgeColor
              emission <- edgeEmission
            yield Some(
              baseMaterial
                .withColorOpt(color)
                .withEmissionOpt(emission)
            )
          case None =>
            Left(s"Unknown edge material preset: '$presetName'. Valid presets: ${Material.presetNames.mkString(", ")}")
      case None =>
        // If edge-color or edge-emission specified without edge-material, create a default material
        for
          color <- edgeColor
          emission <- edgeEmission
        yield
          if color.isDefined || emission.isDefined then
            val baseColor = color.getOrElse(Color(1.0f, 1.0f, 1.0f))
            Some(Material(baseColor, emission = emission.getOrElse(0.0f)))
          else
            None

  private def parseEdgeColor(kvPairs: Map[String, String]): Either[String, Option[Color]] =
    kvPairs.get("edge-color") match
      case Some(colorStr) =>
        Try {
          val hexStr = if colorStr.startsWith("#") then colorStr.substring(1) else colorStr
          Color.fromHex(hexStr)
        }.toEither.left.map { e =>
          s"Invalid edge-color value '$colorStr': ${e.getMessage}. " +
            "Color must be hex format (e.g., edge-color=#00FFFF)"
        }.map(Some(_))
      case None => Right(None)
