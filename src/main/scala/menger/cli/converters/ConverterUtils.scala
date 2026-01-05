package menger.cli.converters

import scala.util.Try

object ConverterUtils:
  def unwrapTryEither[A](t: Try[Either[String, A]]): Either[String, A] =
    t.fold(
      error => Left(error.getMessage),
      identity
    )
