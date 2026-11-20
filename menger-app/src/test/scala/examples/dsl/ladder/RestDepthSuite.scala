package examples.dsl.ladder

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** `RestDepth.measure` is what `RestHeightSuite` checks the stored depths against, for any look. */
class RestDepthSuite extends AnyFlatSpec with Matchers:

  private val stored = 1e-4f
  private val tube = 0.02f

  "RestDepth.measure" should "give the stored depth, for the default look" in:
    for top <- List(4, 5) do
      withClue(s"top $top: ")(RestDepth.measure(top, Look.default) shouldBe Stage.restDepth(top) +- stored)

  it should "reach further down, by at most the tube's growth, for thicker edge tubes" in:
    val thick = Look.default.copy(edgeRadius = Look.default.edgeRadius + tube)
    val (thin, thicker) = (RestDepth.measure(4, Look.default), RestDepth.measure(4, thick))
    thicker should be >= thin - stored
    thicker should be <= thin + tube + stored
    RestDepth.measure(4, Look.default.copy(edgeRadius = 0.3f)) should be > thin + stored

  it should "follow the turns of the look" in:
    val tilted = Look.default.copy(turns = c => phase =>
      menger.spec.PlaneRotation("yz", 35f) :: c.planeAngles(phase))
    val (turned, plain) = (RestDepth.measure(4, tilted), RestDepth.measure(4, Look.default))
    turned should not be plain +- stored

  "Look.withMeasuredRestDepth" should "set the depth its turns need, and the stage rests on it" in:
    val tilted = Look.default.copy(turns = c => phase =>
      menger.spec.PlaneRotation("yz", 35f) :: c.planeAngles(phase)).withMeasuredRestDepth(4)
    tilted.restDepth shouldBe Some(RestDepth.measure(4, tilted))
    val centre = DimensionLadder.sceneAt(0f, top = 4, look = tilted).objects.last.toObjectSpec.y
    centre shouldBe -0.45f + 0.01f + tilted.restDepth.get +- stored
