package menger.engines.scene

import menger.dsl.Vec3

/** OptiX curve primitives render every L-system run through a cubic B-spline
  * (OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE), which approximates control points rather than
  * passing through them. That smoothing is desirable for organic branch bends but rounds off
  * a geometric curve's sharp turns (e.g. hilbert3d's 90-degree corners) into a soft spiral.
  * Tripling a control point at a sharp direction change anchors the spline to that point,
  * producing a visible corner without affecting the smooth majority of the curve.
  */
private[scene] object CurveCornerSharpening:

  /** Empirically separates hilbert3d's 90-degree turns from fern3d/tree's ~25-36-degree
    * organic bends: turns sharper than this angle get anchored, gentler bends stay smooth.
    */
  private val SharpAngleThresholdDegrees: Float = 60f
  private val CosSharpThreshold: Float =
    math.cos(math.toRadians(SharpAngleThresholdDegrees.toDouble)).toFloat
  private val MinDirectionLength: Float = 1e-6f

  def sharpenCorners(points: Vector[Vec3], widths: Vector[Float]): (Vector[Vec3], Vector[Float]) =
    if points.length < 3 then (points, widths)
    else
      val last = points.length - 1
      (1 until last).foldLeft((Vector(points.head), Vector(widths.head))) {
        case ((accP, accW), i) =>
          val withPoint = (accP :+ points(i), accW :+ widths(i))
          if isSharpCorner(points(i - 1), points(i), points(i + 1)) then
            (withPoint._1 ++ Vector.fill(2)(points(i)), withPoint._2 ++ Vector.fill(2)(widths(i)))
          else
            withPoint
      } match
        case (accP, accW) => (accP :+ points(last), accW :+ widths(last))

  private def isSharpCorner(prev: Vec3, corner: Vec3, next: Vec3): Boolean =
    val inDir = corner - prev
    val outDir = next - corner
    val inLen = inDir.magnitude
    val outLen = outDir.magnitude
    if inLen < MinDirectionLength || outLen < MinDirectionLength then false
    else (inDir.normalize dot outDir.normalize) < CosSharpThreshold
