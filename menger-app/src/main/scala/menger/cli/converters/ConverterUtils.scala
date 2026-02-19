package menger.cli.converters

import scala.util.Try

object ConverterUtils:
  def unwrapTryEither[A](t: Try[Either[String, A]]): Either[String, A] =
    t.fold(
      error => Left(error.getMessage),
      identity
    )

  def parseFourDRotation(s: String): Either[String, (Float, Float, Float)] =
    val parts = s.split(",").map(_.trim)
    if parts.length != 3 then
      Left(s"--rotation-4d must be XW,YW,ZW (three comma-separated degrees, e.g., 30,20,0), got: '$s'")
    else
      val results = parts.map { p =>
        try Right(p.toFloat)
        catch case _: NumberFormatException => Left(p)
      }
      results.find(_.isLeft) match
        case Some(Left(bad)) => Left(s"--rotation-4d: '$bad' is not a valid number")
        case _ =>
          val floats = results.collect { case Right(f) => f }
          if floats.exists(f => f < 0 || f >= 360) then
            Left("--rotation-4d values must each be in range [0, 360)")
          else Right((floats(0), floats(1), floats(2)))
