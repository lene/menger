package menger.spec

/**
 * 4D projection parameters for hypercube objects (tesseract, etc.).
 * Only applicable to object types where ObjectType.isHypercube returns true.
 */
case class Projection4DSpec(
  eyeW: Float = Projection4DSpec.DefaultEyeW,
  screenW: Float = Projection4DSpec.DefaultScreenW,
  rotXW: Float = Projection4DSpec.DefaultRotXW,
  rotYW: Float = Projection4DSpec.DefaultRotYW,
  rotZW: Float = Projection4DSpec.DefaultRotZW,
  // Scales the object's w coordinate before rotation and projection (menger#65): 0 flattens a
  // tesseract to a cube, animating 0 -> 1 grows it along w. Geometry, not view: see
  // ObjectSpec.withoutView.
  wScale: Float = Projection4DSpec.DefaultWScale
):
  require(eyeW > screenW, s"eyeW ($eyeW) must be greater than screenW ($screenW)")
  require(eyeW > 0 && screenW > 0, "eyeW and screenW must be positive")
  require(wScale >= 0f && !wScale.isInfinite, s"wScale must be finite and >= 0, got $wScale")

object Projection4DSpec:
  val DefaultEyeW: Float = 3.0f
  val DefaultScreenW: Float = 1.5f
  val DefaultRotXW: Float = 15f
  val DefaultRotYW: Float = 10f
  val DefaultRotZW: Float = 0f
  val DefaultWScale: Float = 1f

  val default: Projection4DSpec = Projection4DSpec()
