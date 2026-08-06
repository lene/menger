package menger.cli.converters

import menger.common.Vector
import org.rogach.scallop.ArgType
import org.rogach.scallop.ValueConverter

given vector3Converter: ValueConverter[Vector[3]] with
  val argType: ArgType.V = ArgType.SINGLE

  def parse(s: List[(String, List[String])]): Either[String, Option[Vector[3]]] =
    if s.isEmpty || s.head._2.isEmpty then Right(None)
    else
      val input = s.head._2.head.trim
      ConverterUtils.parseFloatComponents(input, 3)
        .map(f => Some(Vector[3](f(0), f(1), f(2))))
        .left.map(msg => s"Vector3 '$input': $msg. Expected format: x,y,z (e.g., 0,0,3)")
