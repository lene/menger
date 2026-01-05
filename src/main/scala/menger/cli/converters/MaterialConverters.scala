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
        case e: Exception => Left(s"Color '$input' not recognized: ${e.getMessage}")
      })

  private def parseColorValue(input: String): Either[String, Option[Color]] =
    if input.contains(',') then parseInts(input)
    else parseHex(input)

  private def isValidHexColorLength(len: Int): Boolean = (6 to 8).contains(len)

  private def parseHex(input: String): Either[String, Option[Color]] =
    input.length match
      case len if isValidHexColorLength(len) => Right(Some(Color.valueOf(input)))
      case _ => Left(s"Color '$input' must be a name or a hex value RRGGBB or RRGGBBAA")

  private def hasInvalidCommaPlacement(input: String): Boolean =
    input.startsWith(",") || input.endsWith(",")

  private def isValidRgbValue(n: Int): Boolean =
    n >= 0 && n <= Const.rgbMaxValue

  private def parseInts(input: String): Either[String, Option[Color]] =
    val parts = input.trim.split(",").map(_.trim)
    parts.length match
      case n if hasInvalidCommaPlacement(input) =>
        Left(s"Color '$input' must not start or end with a comma")
      case n if n < 3 || n > 4 =>
        Left(s"Color '$input' must have 3 or 4 components")
      case _ =>
        val nums = parts.map(_.toInt)
        if !nums.forall(isValidRgbValue) then
          Left(s"Color '$input' has values out of range 0-${Const.rgbMaxValue}")
        else
          Right(Some(ColorConversions.rgbIntsToColor(nums)))
