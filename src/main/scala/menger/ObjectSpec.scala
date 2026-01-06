package menger

import scala.util.Try

import menger.common.Color
import menger.common.ObjectType
import menger.optix.Material

/**
 * Object specification parsed from CLI --object flag.
 * Format: type=TYPE:key=value:key=value...
 *
 * Examples:
 *   type=sphere:pos=0,0,0:size=1.0:color=#FF0000:ior=1.5
 *   type=cube:pos=2,0,0:size=1.5:color=#0000FF
 *   type=sponge-volume:pos=-2,0,0:size=2.0:level=3:color=#00FF00
 *   type=sponge-surface:pos=0,2,0:size=2.0:level=2.5:color=#FFFF00
 *   type=cube-sponge:pos=1,0,0:size=2.0:level=2:color=#FF00FF
 *   type=sphere:pos=0,0,0:material=glass:ior=1.7
 *   type=cube:pos=0,0,0:material=metal:color=#FFD700
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
  material: Option[Material] = None
)

object ObjectSpec:

  /**
   * Parse object specification from keyword=value format.
   * Format: type=TYPE:pos=x,y,z:size=S:level=L:color=#RRGGBB:ior=I:material=PRESET
   * 
   * Material keywords:
   *   material=PRESET - base material preset (glass, water, diamond, chrome, gold, copper, metal, plastic, matte)
   *   ior=VALUE       - override IOR (also applies without material preset)
   *   roughness=VALUE - override roughness (only with material preset)
   *   metallic=VALUE  - override metallic (only with material preset)
   *   specular=VALUE  - override specular (only with material preset)
   *   color=#RRGGBB   - override color (works with or without material preset)
   */
  def parse(spec: String): Either[String, ObjectSpec] =
    val parts = spec.split(":").map(_.trim)
    val kvPairs = parts.flatMap { part =>
      part.split("=", 2) match
        case Array(key, value) => Some(key.trim.toLowerCase -> value.trim)
        case _ => None
    }.toMap

    for
      objType <- parseObjectType(kvPairs)
      (x, y, z) <- parsePosition(kvPairs)
      size <- parseSize(kvPairs)
      level <- parseLevel(kvPairs)
      color <- parseColor(kvPairs)
      ior <- parseIOR(kvPairs)
      material <- parseMaterial(kvPairs, color, ior)
      _ <- validateSpongeLevel(objType, level)
    yield ObjectSpec(objType, x, y, z, size, level, color, ior, material)

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
            Try((xStr.toFloat, yStr.toFloat, zStr.toFloat)).toEither.left.map(_.getMessage)
          case _ =>
            Left(s"Invalid position format: '$posStr'. Expected format: pos=x,y,z " +
              "(three comma-separated numbers). Example: pos=1.0,2.0,3.0")
      case None => Right((0.0f, 0.0f, 0.0f))

  private def parseSize(kvPairs: Map[String, String]): Either[String, Float] =
    Try(kvPairs.get("size").map(_.toFloat).getOrElse(1.0f)).toEither.left.map(_.getMessage)

  private def parseLevel(kvPairs: Map[String, String]): Either[String, Option[Float]] =
    Try(kvPairs.get("level").map(_.toFloat)).toEither.left.map(_.getMessage)

  private def parseColor(kvPairs: Map[String, String]): Either[String, Option[menger.common.Color]] =
    Try {
      kvPairs.get("color").map { colorStr =>
        val hexStr = if colorStr.startsWith("#") then colorStr.substring(1) else colorStr
        menger.common.Color.fromHex(hexStr)
      }
    }.toEither.left.map(_.getMessage)

  private def parseIOR(kvPairs: Map[String, String]): Either[String, Float] =
    Try(kvPairs.get("ior").map(_.toFloat).getOrElse(1.0f)).toEither.left.map(_.getMessage)

  private def parseMaterial(
    kvPairs: Map[String, String],
    color: Option[Color],
    ior: Float
  ): Either[String, Option[Material]] =
    kvPairs.get("material") match
      case Some(presetName) =>
        Material.fromName(presetName) match
          case Some(baseMaterial) =>
            Try {
              val withColor = color.map(c => baseMaterial.copy(color = c)).getOrElse(baseMaterial)
              val withIor = if kvPairs.contains("ior") then withColor.copy(ior = ior) else withColor
              val withRoughness = kvPairs.get("roughness")
                .map(v => withIor.copy(roughness = v.toFloat))
                .getOrElse(withIor)
              val withMetallic = kvPairs.get("metallic")
                .map(v => withRoughness.copy(metallic = v.toFloat))
                .getOrElse(withRoughness)
              val withSpecular = kvPairs.get("specular")
                .map(v => withMetallic.copy(specular = v.toFloat))
                .getOrElse(withMetallic)
              Some(withSpecular)
            }.toEither.left.map(e => s"Invalid material override: ${e.getMessage}")
          case None =>
            Left(s"Unknown material preset: '$presetName'. Valid presets: ${Material.presetNames.mkString(", ")}")
      case None =>
        Right(None)

  private def validateSpongeLevel(objType: String, level: Option[Float]): Either[String, Unit] =
    if ObjectType.isSponge(objType) && level.isEmpty then
      Left("Sponge object requires 'level' field. Add level=<number> to specification. " +
        s"Example: type=$objType:level=2")
    else
      Right(())

  /**
   * Validate multiple object specifications.
   * Returns error if any spec is invalid.
   */
  def parseAll(specs: List[String]): Either[String, List[ObjectSpec]] =
    specs.foldLeft[Either[String, List[ObjectSpec]]](Right(List.empty)) { (acc, spec) =>
      acc.flatMap { objects =>
        parse(spec).map(obj => objects :+ obj)
      }
    }
