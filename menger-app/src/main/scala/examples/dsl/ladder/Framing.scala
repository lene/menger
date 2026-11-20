package examples.dsl.ladder

/** What the camera keeps inside the frame: the object's `Bound`, an upper bound for any turn of
  * it (`Choreography.boundingRadius`), or its `Extent`, the projected distance of its farthest
  * corner in the frame being drawn (`Choreography.halfExtent`), which is tighter. */
enum Fit:
  case Bound, Extent

/** How the camera frames the scene (menger#194): the renderer's horizontal field of view in
  * degrees, the `margin` of the frame's half width and half height the fitted points may use, and
  * what is `fit`ted of the object. The defaults are the 45 degree field, the 0.95 margin and the
  * bound the ladder always had. */
final case class Framing(
  fovDegrees: Float = Framing.defaultFov,
  margin: Float = Framing.defaultMargin,
  fit: Fit = Fit.Bound
):
  require(fovDegrees > 0f && fovDegrees < Framing.straightAngle,
    s"the field of view must be between 0 and ${Framing.straightAngle} degrees, got $fovDegrees")
  require(margin > 0f && margin <= 1f, s"the margin must be above 0 and at most 1, got $margin")

  /** The tangent of half the field of view, to the four decimals the camera always used, so that
    * the default framing leaves the default scene bit for bit as it was. */
  val tanHalfFovX: Float =
    math.round(math.tan(math.toRadians(fovDegrees / 2.0)) * Framing.tangentScale).toFloat /
      Framing.tangentScale.toFloat

object Framing:
  val defaultFov = 45f
  val defaultMargin = 0.95f
  private val straightAngle = 180f
  private val tangentScale = 10000.0

  val default: Framing = Framing()
