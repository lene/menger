package menger.cli.converters

import menger.dsl.Ambient
import org.rogach.scallop.ArgType
import org.rogach.scallop.ValueConverter

given ambientConverter: ValueConverter[Ambient] with
  val argType: ArgType.V = ArgType.SINGLE

  def parse(s: List[(String, List[String])]): Either[String, Option[Ambient]] =
    if s.isEmpty || s.head._2.isEmpty then Right(None)
    else Ambient.parse(s.head._2.head).map(Some(_))
