package menger.objects.higher_d

import menger.common.Vector

/** A 4D point's way to 3D through higher dimensions (menger#162): it is lifted into
  * `dimension` by zero coordinates, turned by `turns` in written order (the first turns first),
  * and projected down by a [[PerspectiveChain]] with one stage per dropped axis, the first stage
  * dropping the highest axis. `dimension` 4 is the plain 4D view. */
final class LiftedProjection private (
  val dimension: Int,
  project4D: Vector[4] => Vector[3],
  clip4D: Vector[4] => Option[Vector[3]]
):

  /** Unclipped, like `Projection.apply`: a point at or behind a stage's eye is not valid. */
  def apply(point: Vector[4]): Vector[3] = project4D(point)

  /** `None` for a point at or behind a stage's eye. */
  def project(point: Vector[4]): Option[Vector[3]] = clip4D(point)

object LiftedProjection:
  private val SourceDimension = 4

  /** The 4D coordinates followed by zeros up to `dimension`. */
  def lift(point: Vector[4], dimension: Int): IndexedSeq[Float] =
    IndexedSeq.tabulate(dimension)(axis => if axis < SourceDimension then point(axis) else 0f)

  def apply(
    dimension: Int, turns: Seq[(Plane, Float)], stages: Seq[Projection]
  ): LiftedProjection =
    require(dimension >= SourceDimension, s"cannot lift a 4D point into dimension $dimension")
    val d = Dimension.of(dimension)
    given ValueOf[d.N] = d.valueOf
    val rotation = Rotation.inWrittenOrder[d.N](turns)
    val chain = PerspectiveChain[d.N](stages)
    def lifted(point: Vector[4]): Vector[d.N] = Vector[d.N](lift(point, dimension)*)
    new LiftedProjection(
      dimension,
      point => chain(rotation(lifted(point))),
      point => chain.project(rotation(lifted(point)))
    )
