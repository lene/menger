package menger.common

// Placed in menger.common rather than menger.config because menger.input (its primary
// consumer) is in menger-app, which already has access to menger.common.  Moving it to
// menger.config would work technically, but menger.common keeps input-tuning parameters
// alongside the other shared primitives (Color, Vector, Material) and avoids a superfluous
// intra-module dependency on menger.config solely for one type.
case class OrbitConfig(
  orbitSensitivity: Float = Const.Input.defaultZoomSensitivity,
  panSensitivity: Float = Const.Input.defaultPanSensitivity,
  zoomSensitivity: Float = Const.Input.zoomSensitivity,
  minDistance: Float = Const.Input.defaultMinDistance,
  maxDistance: Float = Const.Input.defaultMaxDistance,
  minElevation: Float = Const.Input.defaultMinElevation,
  maxElevation: Float = Const.Input.defaultMaxElevation
)
