package menger.config

import menger.common.PlaneColorSpec
import menger.common.PlaneSpec
import menger.optix.Material

/** Rendering configuration for a single plane (spec + coloring + optional material).
  *
  * Lives in menger.config because it combines a domain PlaneSpec (common) with
  * a rendering Material (optix), making it a config-layer bridge type.
  */
case class PlaneConfig(
  spec: PlaneSpec,
  colorSpec: Option[PlaneColorSpec],
  material: Option[Material] = None
)
