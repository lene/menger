package menger.config

import menger.common.Material

case class CrossConfig(
  enabled: Boolean = false,
  length: Float = 2.0f,
  thickness: Float = 0.03f,
  material: Option[Material] = None
)

object CrossConfig:
  val Disabled: CrossConfig = CrossConfig()
