package menger.dsl

case class IBL(
  strength: Float = 1.0f,
  samples:  Int   = 1,
):
  require(samples >= 1 && samples <= 8, s"IBL.samples must be 1–8, got $samples")
  require(strength >= 0.0f,            s"IBL.strength must be ≥ 0, got $strength")
