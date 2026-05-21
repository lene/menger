package menger.cli.converters

import scala.util.Try

import menger.cli.FogSpec
import menger.common.Color
import org.rogach.scallop.ArgType
import org.rogach.scallop.ValueConverter

given fogSpecConverter: ValueConverter[FogSpec] with
  val argType: ArgType.V = ArgType.SINGLE

  def parse(s: List[(String, List[String])]): Either[String, Option[FogSpec]] =
    if s.isEmpty || s.head._2.isEmpty then Right(None)
    else
      val input = s.head._2.head.trim
      Try(parseFogSpec(input)).toEither.left
        .map(e => s"Fog spec '$input' not recognized: ${e.getMessage}. " +
          "Expected format: density=0.05:color=0.8,0.8,0.9")
        .map(Some(_))

  private def parseFogSpec(input: String): FogSpec =
    val parts = input.split(':').map(_.trim).toList
    val density = parts
      .find(_.startsWith("density="))
      .map(_.drop(8).toFloat)
      .getOrElse(0.05f)
    val color = parts
      .find(_.startsWith("color="))
      .map { part =>
        val rgb = part.drop(6).split(',')
        if rgb.length == 3 then Color(rgb(0).toFloat, rgb(1).toFloat, rgb(2).toFloat)
        else Color(0.8f, 0.8f, 0.9f)
      }
      .getOrElse(Color(0.8f, 0.8f, 0.9f))
    FogSpec(density, color)
