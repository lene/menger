package menger.cli.converters

import scala.util.Try

object ConverterUtils:
  def unwrapTryEither[A](t: Try[Either[String, A]]): Either[String, A] =
    t.fold(
      error => Left(error.getMessage),
      identity
    )

  def parseFloatComponents(input: String, expectedCount: Int): Either[String, Array[Float]] =
    val parts = input.split(",").map(_.trim)
    if parts.length != expectedCount then
      Left(s"expected $expectedCount comma-separated values, got ${parts.length} in '$input'")
    else
      val results = parts.map(p => Try(p.toFloat).toEither.left.map(_ => p))
      results.find(_.isLeft) match
        case Some(Left(bad)) => Left(s"'$bad' is not a valid number in '$input'")
        case _               => Right(results.collect { case Right(f) => f })

  def parseFourDRotation(s: String): Either[String, (Float, Float, Float)] =
    parseFloatComponents(s, 3).flatMap { floats =>
      if floats.exists(f => f < 0 || f >= 360) then
        Left("--rotation-4d values must each be in range [0, 360)")
      else
        Right((floats(0), floats(1), floats(2)))
    }.left.map {
      case msg if msg.startsWith("expected") =>
        s"--rotation-4d must be XW,YW,ZW (three comma-separated degrees, e.g., 30,20,0), got: '$s'"
      case msg => s"--rotation-4d: $msg"
    }
