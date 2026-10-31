package menger.dsl

case class IBL(
  strength: Float = 1.0f,
  samples:  Int   = 1,
):
  FieldConstraints.check("IBL", "samples", samples)
  require(strength >= 0.0f,            s"IBL.strength must be ≥ 0, got $strength")
