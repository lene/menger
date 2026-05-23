package menger.dsl

sealed trait ToneMapping
object ToneMapping:
  case object None extends ToneMapping
  case class Reinhard(exposure: Float = 1.0f) extends ToneMapping
  case class ACES(exposure: Float = 1.0f) extends ToneMapping
