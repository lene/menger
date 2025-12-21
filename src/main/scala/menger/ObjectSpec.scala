package menger

import scala.util.Try

import menger.common.Color
import menger.common.ObjectType

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
 */
case class ObjectSpec(
  objectType: String,
  x: Float = 0.0f,
  y: Float = 0.0f,
  z: Float = 0.0f,
  size: Float = 1.0f,
  level: Option[Float] = None,
  color: Option[Color] = None,
  ior: Float = 1.0f
)

object ObjectSpec:

  /**
   * Parse object specification from keyword=value format.
   * Format: type=TYPE:pos=x,y,z:size=S:level=L:color=#RRGGBB:ior=I
   */
  def parse(spec: String): Either[String, ObjectSpec] =
    val parts = spec.split(":").map(_.trim)
    val kvPairs = parts.flatMap { part =>
      part.split("=", 2) match
        case Array(key, value) => Some(key.trim.toLowerCase -> value.trim)
        case _ => None
    }.toMap

    for
      // Required: type
      objType <- kvPairs.get("type") match
        case Some(t) if ObjectType.isValid(t) =>
          Right(t.toLowerCase)
        case Some(t) =>
          Left(s"Invalid object type: $t (valid: ${ObjectType.validTypesString})")
        case None =>
          Left("Missing required 'type' field")

      // Optional: position (pos=x,y,z)
      position <- kvPairs.get("pos") match
        case Some(posStr) =>
          posStr.split(",").map(_.trim) match
            case Array(px, py, pz) =>
              Try((px.toFloat, py.toFloat, pz.toFloat)).toEither.left.map(_.getMessage)
            case _ =>
              Left(s"Invalid position format: $posStr (expected x,y,z)")
        case None => Right((0.0f, 0.0f, 0.0f))
      (x, y, z) = position

      // Optional: size
      size <- Try(kvPairs.get("size").map(_.toFloat).getOrElse(1.0f)).toEither.left.map(_.getMessage)

      // Optional: level (for sponges)
      level <- Try(kvPairs.get("level").map(_.toFloat)).toEither.left.map(_.getMessage)

      // Optional: color
      color <- Try {
        kvPairs.get("color").map { colorStr =>
          val hexStr = if colorStr.startsWith("#") then colorStr.substring(1) else colorStr
          Color.fromHex(hexStr)
        }
      }.toEither.left.map(_.getMessage)

      // Optional: ior
      ior <- Try(kvPairs.get("ior").map(_.toFloat).getOrElse(1.0f)).toEither.left.map(_.getMessage)

      // Validate: sponges should have level
      _ <- if ObjectType.isSponge(objType) && level.isEmpty then
        Left("Sponge object requires 'level' field")
      else
        Right(())

    yield ObjectSpec(objType, x, y, z, size, level, color, ior)

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
