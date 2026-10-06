package menger.engines

import menger.config.TAnimationConfig
import menger.dsl.Camera
import menger.dsl.LoadedScene
import menger.dsl.Scene
import menger.dsl.Sphere
import menger.dsl.Vec3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PreviewEngineReloadSuite extends AnyFlatSpec with Matchers:

  private val current = TAnimationConfig(startT = 0f, endT = 10f, frames = 100, savePattern = "")
  private val aScene = Scene(
    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
    objects = List(Sphere(pos = Vec3(0f, 0f, 0f))),
    lights = List()
  )
  private val sceneFn: Float => Scene = _ => aScene
  private def animated(duration: Option[Float]) = LoadedScene.Animated(sceneFn)(duration)

  "PreviewEngine.reloadDecision" should "adopt the new duration of a real-time scene" in:
    PreviewEngine.reloadDecision(Right(animated(Some(20f))), current, realtime = true) match
      case PreviewEngine.Reload(_, config) => config.endT shouldBe 20f
      case other                           => fail(s"expected a reload, got $other")

  it should "keep the time range when a real-time scene drops its duration" in:
    PreviewEngine.reloadDecision(Right(animated(None)), current, realtime = true) match
      case PreviewEngine.Reload(_, config) => config shouldBe current
      case other                           => fail(s"expected a reload, got $other")

  it should "keep the range a --preview scene was started with" in:
    PreviewEngine.reloadDecision(Right(animated(Some(20f))), current, realtime = false) match
      case PreviewEngine.Reload(_, config) => config shouldBe current
      case other                           => fail(s"expected a reload, got $other")

  it should "not reload a scene that became static" in:
    val loaded = Right(LoadedScene.Static(aScene))
    PreviewEngine.reloadDecision(loaded, current, realtime = true) shouldBe PreviewEngine.KindChanged

  it should "reject a scene that failed to load and keep the playing one" in:
    PreviewEngine.reloadDecision(Left("compile error"), current, realtime = true) shouldBe
      PreviewEngine.Rejected("compile error")
