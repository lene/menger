package examples.dsl.ladder

import menger.spec.ObjectSpec

/** How far below its centre the object reaches over a whole cycle, for any look: the number
  * `Stage.restDepths` stores for the default look, found the way `RestHeightSuite` checks it.
  * A look with other turns or thicker edge tubes may reach lower, and the stage must then rest
  * the object higher. */
object RestDepth:
  private val framesPerSecond = 30

  // A sponge lies inside its n-cube, and a projection keeps the hull: its level-0 mesh is the hull.
  private def hullOf(spec: ObjectSpec): ObjectSpec = spec.copy(level = spec.level.map(_ => 0f))

  private def reachOf(spec: ObjectSpec): Float =
    val lowest = WorldMesh.corners(hullOf(spec)).map(_._2).min
    spec.y - lowest + spec.edgeRadius.getOrElse(0f)

  /** The largest reach below the centre over every frame at 30 fps and both ends of every
    * segment of the cycle, of the n-cube of each frame (its lowest corner) and its edge tubes. */
  def measure(top: Int, look: Look = Look.default,
      cycle: Float = DimensionLadder.cycleSeconds): Float =
    val sampled = (0 until (cycle * framesPerSecond).toInt).map: i =>
      DimensionLadder.sceneAt(i.toFloat / framesPerSecond, cycle, top, look).objects.last
    val ends = DimensionLadder.segments(cycle, top).flatMap: segment =>
      List(0f, 1f).map(u => DimensionLadder.objectAt(segment, u, cycle, top, look))
    (sampled ++ ends).map(obj => reachOf(obj.toObjectSpec)).max
