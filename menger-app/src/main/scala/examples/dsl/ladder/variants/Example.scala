package examples.dsl.ladder.variants

import examples.dsl.ladder.Choreography
import examples.dsl.ladder.Choreography.Kind
import examples.dsl.ladder.DimensionLadder
import examples.dsl.ladder.Look
import menger.dsl.Material
import menger.dsl.Scene

/** The shape of a variant (menger#167): `Look.default` with one thing changed, here chrome for
  * the metal, shown at three moments by column: `--scene examples.dsl.ladder.variants.Example
  * --at 0,1,2` is the hold, the middle of the 3D breath and the middle of the 4D fold. */
object Example:
  val moments: Vector[Moment] =
    Vector(Moment(Kind.Hold, 3, 0f), Moment(Kind.Breath, 3, 0.5f), Moment(Kind.Fold, 4, 0.5f))

  private def look: Look = Look.default.copy(metal = Material.Chrome)

  private def timeline: Choreography =
    Choreography(DimensionLadder.cycleSeconds, DimensionLadder.topDimension, look.depths)

  def scene(t: Float): Scene =
    DimensionLadder.sceneAt(Moment.ofColumn(timeline, moments)(t), look = look)
