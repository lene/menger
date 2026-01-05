package menger.cli.converters

import scala.util.Try

import com.badlogic.gdx.math.Vector3
import menger.cli.converters.ConverterUtils.unwrapTryEither
import org.rogach.scallop.ArgType
import org.rogach.scallop.ValueConverter

given vector3Converter: ValueConverter[Vector3] with
  val argType: ArgType.V = ArgType.SINGLE

  def parse(s: List[(String, List[String])]): Either[String, Option[Vector3]] =
    if s.isEmpty || s.head._2.isEmpty then Right(None)
    else
      val input = s.head._2.head.trim
      unwrapTryEither(Try {
        val parts = input.split(",").map(_.trim.toFloat)
        if parts.length != 3 then
          Left(s"Vector3 '$input' must have exactly 3 components (x,y,z)")
        else
          Right(Some(Vector3(parts(0), parts(1), parts(2))))
      }.recover {
        case e: NumberFormatException => Left(s"Vector3 '$input' contains non-numeric values")
        case e: Exception => Left(s"Vector3 '$input' not recognized: ${e.getMessage}")
      })
