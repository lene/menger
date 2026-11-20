package examples.dsl.ladder.variants

import examples.dsl.ladder.Choreography
import examples.dsl.ladder.Choreography.Kind

/** A point of the timeline by what it shows: `Moment(Kind.Breath, 5, 0.5f)` is the middle of the
  * 5D breath, `u` of the way through the segment. */
final case class Moment(kind: Kind, dimension: Int, u: Float)

/** Variants with another top dimension or other depths have other timelines, so the same second
  * shows different things. A sheet compares them by moment instead. */
object Moment:

  /** The time of `u` of the way through the segment of `kind` and `dimension`. At u = 1 it is the
    * last instant before the next segment begins, so that the segment found there is still the
    * one asked for. */
  def of(choreography: Choreography, kind: Kind, dimension: Int, u: Float): Float =
    require(u >= 0f && u <= 1f, s"u must be from 0 to 1, got $u")
    val index = choreography.segments.indexWhere(s => s.kind == kind && s.dimension == dimension)
    require(index >= 0,
      s"the timeline has no $kind segment of dimension $dimension; it has " +
        choreography.segments.map(s => s"${s.kind}(${s.dimension})").mkString(", "))
    val segment = choreography.segments(index)
    val next = choreography.segments.lift(index + 1).fold(choreography.cycle)(_.start)
    val time = segment.start + u * segment.seconds
    if time < next then time else math.nextDown(next)

  /** The time of the moment a sheet's column shows: column 0, 1, 2, ... of `moments`, as the
    * sheet passes them as `--at 0,1,2`. */
  def ofColumn(choreography: Choreography, moments: IndexedSeq[Moment])(column: Float): Float =
    require(column >= 0f && column < moments.length && column == column.floor,
      s"there is no column $column; the sheet has ${moments.length}")
    val moment = moments(column.toInt)
    of(choreography, moment.kind, moment.dimension, moment.u)
