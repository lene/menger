package menger.objects.higher_d

import menger.common.Vector

/** A point's way to 3D through higher dimensions (menger#162): it is lifted into `dimension` by
  * zero coordinates, turned by `turns` in written order (the first turns first), and projected
  * down by a [[PerspectiveChain]] with one stage per dropped axis, the first stage dropping the
  * highest axis. A point of any dimension up to `dimension` is lifted (menger#163: the n-cube);
  * `dimension` 4 is the plain 4D view. */
final class LiftedProjection private (
  val dimension: Int,
  projectLifted: IndexedSeq[Float] => Vector[3],
  clipLifted: IndexedSeq[Float] => Option[Vector[3]]
):

  /** Unclipped, like `Projection.apply`: a point at or behind a stage's eye is not valid. */
  def apply(point: Vector[4]): Vector[3] = apply(point.toIndexedSeq)

  def apply(coordinates: IndexedSeq[Float]): Vector[3] =
    projectLifted(LiftedProjection.lift(coordinates, dimension))

  /** `None` for a point at or behind a stage's eye. */
  def project(point: Vector[4]): Option[Vector[3]] = project(point.toIndexedSeq)

  def project(coordinates: IndexedSeq[Float]): Option[Vector[3]] =
    clipLifted(LiftedProjection.lift(coordinates, dimension))

object LiftedProjection:
  private val SourceDimension = 4

  /** The 4D coordinates followed by zeros up to `dimension`. */
  def lift(point: Vector[4], dimension: Int): IndexedSeq[Float] =
    lift(point.toIndexedSeq, dimension)

  /** The coordinates followed by zeros up to `dimension`. */
  def lift(coordinates: IndexedSeq[Float], dimension: Int): IndexedSeq[Float] =
    require(
      coordinates.size <= dimension,
      s"cannot lift a ${coordinates.size}D point into dimension $dimension"
    )
    coordinates.padTo(dimension, 0f)

  def apply(
    dimension: Int, turns: Seq[(Plane, Float)], stages: Seq[Projection]
  ): LiftedProjection =
    require(dimension >= SourceDimension, s"cannot lift a 4D point into dimension $dimension")
    val d = Dimension.of(dimension)
    given ValueOf[d.N] = d.valueOf
    val rotation = Rotation.inWrittenOrder[d.N](turns)
    val chain = PerspectiveChain[d.N](stages)
    def turned(lifted: IndexedSeq[Float]): Vector[d.N] = rotation(Vector[d.N](lifted*))
    new LiftedProjection(
      dimension,
      lifted => chain(turned(lifted)),
      lifted => chain.project(turned(lifted))
    )
