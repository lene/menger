package menger.dsl

/** Low-level cubic Bezier curve evaluated at t ∈ [0,1].
 *
 *  B(t) = (1-t)³·P0 + 3(1-t)²t·P1 + 3(1-t)t²·P2 + t³·P3
 *
 *  B(0) = P0, B(1) = P3. P1 and P2 are tangent handles (off the curve).
 */
object Bezier:
  def cubic(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3)(t: Float): Vec3 =
    val u   = 1f - t
    val uu  = u * u
    val tt  = t * t
    p0 * (uu * u) + p1 * (3f * uu * t) + p2 * (3f * u * tt) + p3 * (tt * t)


/** Smooth camera path through waypoints using a Catmull-Rom spline.
 *
 *  Unlike Bezier, Catmull-Rom passes *through* every control point, so
 *  positions are simply the waypoints the camera visits at t = 0, 1/n, 2/n, …, 1.
 *  Phantom endpoints are added internally to give natural boundary tangents.
 *
 *  @param positions At least 2 waypoint positions.
 *  @param lookAt    Fixed look-at target used for all cameras on the path.
 *  @param up        Camera up vector.
 */
case class CameraPath(
  positions: List[Vec3],
  lookAt: Vec3 = Vec3.Zero,
  up: Vec3 = Vec3(0f, 1f, 0f)
):
  require(positions.size >= 2, "CameraPath requires at least 2 positions")

  /** Return the Camera at global parameter t ∈ [0,1]. Clamped at boundaries. */
  def at(t: Float): Camera =
    Camera(position = positionAt(t), lookAt = lookAt, up = up)

  /** Return just the eye position at global parameter t ∈ [0,1]. */
  def positionAt(t: Float): Vec3 =
    val clamped  = math.max(0f, math.min(1f, t))
    val segments = positions.size - 1
    val scaled   = clamped * segments
    val seg      = math.min(scaled.toInt, segments - 1)
    catmullRom(seg, scaled - seg)

  // Phantom points: repeat first and last so boundary segments have defined tangents.
  private lazy val pts: Vector[Vec3] =
    (positions.head +: positions :+ positions.last).toVector

  private def catmullRom(seg: Int, t: Float): Vec3 =
    val p0  = pts(seg)
    val p1  = pts(seg + 1)
    val p2  = pts(seg + 2)
    val p3  = pts(seg + 3)
    val tt  = t * t
    val ttt = tt * t
    p0 * (-0.5f * ttt +        tt - 0.5f * t      ) +
    p1 * ( 1.5f * ttt - 2.5f * tt              + 1f) +
    p2 * (-1.5f * ttt + 2.0f * tt + 0.5f * t      ) +
    p3 * ( 0.5f * ttt - 0.5f * tt                 )


object CameraPath:
  /** Convenience constructor: path through positions, always looking at a fixed target. */
  def lookingAt(lookAt: Vec3, positions: Vec3*): CameraPath =
    CameraPath(positions.toList, lookAt = lookAt)
