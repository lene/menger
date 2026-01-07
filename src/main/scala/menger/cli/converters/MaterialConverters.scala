package menger.cli.converters

import scala.util.Try

import com.badlogic.gdx.graphics.Color
import menger.ColorConversions
import menger.cli.converters.ConverterUtils.unwrapTryEither
import menger.common.Const
import org.rogach.scallop.ArgType
import org.rogach.scallop.ValueConverter

given colorConverter: ValueConverter[Color] with
  val argType: ArgType.V = ArgType.SINGLE

  def parse(s: List[(String, List[String])]): Either[String, Option[Color]] =
    if s.isEmpty || s.head._2.isEmpty then Right(None)
    else
      val input = s.head._2.head.trim
      unwrapTryEither(Try(parseColorValue(input)).recover {
        case e: Exception =>
          Left(s"Color '$input' not recognized: ${e.getMessage}. " +
            "Expected hex (RRGGBB/RRGGBBAA) or RGB (r,g,b or r,g,b,a with values 0-255)")
      })

  private def parseColorValue(input: String): Either[String, Option[Color]] =
    if input.contains(',') then parseInts(input)
    else parseHex(input)

  private def isValidHexColorLength(len: Int): Boolean = (6 to 8).contains(len)

  private def parseHex(input: String): Either[String, Option[Color]] =
    input.length match
      case len if isValidHexColorLength(len) => Right(Some(Color.valueOf(input)))
      case _ =>
        Left(s"Color '$input' must be hex format RRGGBB (6 digits) or RRGGBBAA (8 digits). " +
          "Example: FF0000 (red) or 00FF0080 (green with 50% alpha)")

  private def hasInvalidCommaPlacement(input: String): Boolean =
    input.startsWith(",") || input.endsWith(",")

  private def isValidRgbValue(n: Int): Boolean =
    n >= 0 && n <= Const.rgbMaxValue

  private def parseInts(input: String): Either[String, Option[Color]] =
    val parts = input.trim.split(",").map(_.trim)
    parts.length match
      case n if hasInvalidCommaPlacement(input) =>
        Left(s"Color '$input' has invalid format: must not start or end with a comma. " +
          "Example: 255,0,0 or 255,128,0,200")
      case n if n < 3 || n > 4 =>
        Left(s"Color '$input' must have 3 components (R,G,B) or 4 components (R,G,B,A). " +
          "Example: 255,0,0 or 255,128,0,200")
      case _ =>
        val nums = parts.map(_.toInt)
        if !nums.forall(isValidRgbValue) then
          Left(s"Color '$input' has values out of range. " +
            s"All components must be between 0 and ${Const.rgbMaxValue}")
        else
          Right(Some(ColorConversions.rgbIntsToColor(nums)))
