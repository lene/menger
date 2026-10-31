package menger.cli.converters

import scala.util.Try

import menger.cli.ObjectSpecParser
import menger.cli.converters.ConverterUtils.unwrapTryEither
import menger.config.AnimationSpecificationSequence
import menger.spec.ObjectSpec
import org.rogach.scallop.ArgType
import org.rogach.scallop.ValueConverter

given animationSpecificationSequenceConverter: ValueConverter[AnimationSpecificationSequence] with
  val argType: ArgType.V = ArgType.LIST

  def parse(s: List[(String, List[String])]): Either[String, Option[AnimationSpecificationSequence]] =
    val specStrings = s.flatMap(_(1))
    if specStrings.isEmpty then Right(None)
    else
      unwrapTryEither(Try(Right(Some(AnimationSpecificationSequence(specStrings)))).recover {
        case e: Exception => Left(e.getMessage)
      })

given objectSpecConverter: ValueConverter[List[ObjectSpec]] with
  val argType: ArgType.V = ArgType.LIST

  def parse(s: List[(String, List[String])]): Either[String, Option[List[ObjectSpec]]] =
    val specStrings = s.flatMap(_._2)
    if specStrings.isEmpty then Right(None)
    else
      ObjectSpecParser.parseAll(specStrings) match
        case Right(objects) => Right(Some(objects))
        case Left(error) => Left(error)
