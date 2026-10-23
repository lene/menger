package menger.engines

import menger.ObjectSpec
import menger.common.CausticsConfig
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.config.ExecutionConfig
import menger.config.TAnimationConfig
import menger.dsl.Camera
import menger.dsl.Scene
import menger.dsl.Sphere
import menger.dsl.Vec3
import menger.engines.scene.TesseractEdgeSceneBuilder
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AnimationEngineSuite extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig.disabled

  private val aScene = Scene(
    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
    objects = List(Sphere(pos = Vec3(0f, 0f, 0f))),
    lights = List()
  )
  // 1 face mesh + 96 edge cylinders: more than the default budget of 64.
  private val edged24Cell = List(
    ObjectSpec.parse("type=24-cell:size=0.8:edge-material=gold:edge-radius=0.02").toOption.get
  )

  private class Budget(userSet: Boolean) extends AnimationEngine(
    sceneFunction = _ => aScene,
    animConfig = TAnimationConfig(startT = 0f, endT = 1f, frames = 2, savePattern = "f-%d.png"),
    executionConfig = ExecutionConfig(),
    renderConfig = RenderConfig.Default,
    causticsConfig = CausticsConfig.Disabled,
    userSetMaxInstances = userSet
  ):
    def forEdges: Int =
      computeEffectiveMaxInstances(TesseractEdgeSceneBuilder("."), edged24Cell)

  "AnimationEngine" should "size its instance budget for an edge scene over the default" in:
    Budget(userSet = false).forEdges shouldBe 2 * 97

  it should "keep an explicit --max-instances" in:
    Budget(userSet = true).forEdges shouldBe 64
