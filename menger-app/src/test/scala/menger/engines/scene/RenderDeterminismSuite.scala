package menger.engines.scene

import menger.ObjectSpec
import menger.config.SceneConfig
import org.scalatest.Tag
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

object GPURequired extends Tag("menger.tags.GPURequired")

class RenderDeterminismSuite extends AnyFlatSpec with Matchers:

  private def spec(s: String): ObjectSpec = ObjectSpec.parse(s).toOption.get

  // ── Scene configuration determinism ────────────────────────────────────────

  "SceneConfig" should "be deterministic for identical specs" in:
    val specs1 = List(ObjectSpec("sphere"), ObjectSpec("cube"))
    val specs2 = List(ObjectSpec("sphere"), ObjectSpec("cube"))
    val config1 = SceneConfig.multiObject(specs1)
    val config2 = SceneConfig.multiObject(specs2)
    config1 shouldBe config2

  it should "be deterministic for identical 4D projection specs" in:
    val s1 = spec("type=tesseract:level=2:rot-xw=30:rot-yw=15:eye-w=4:screen-w=2")
    val s2 = spec("type=tesseract:level=2:rot-xw=30:rot-yw=15:eye-w=4:screen-w=2")
    val config1 = SceneConfig.multiObject(List(s1))
    val config2 = SceneConfig.multiObject(List(s2))
    config1 shouldBe config2

  it should "differ for objects with different positions" in:
    val s1 = spec("type=sphere:pos=0,0,0")
    val s2 = spec("type=sphere:pos=1,0,0")
    val config1 = SceneConfig.multiObject(List(s1))
    val config2 = SceneConfig.multiObject(List(s2))
    config1 should not be config2

  // ── GPU render determinism (placeholder) ───────────────────────────────────

  "Render pipeline" should "produce deterministic output for identical scene and seed" taggedAs(
    GPURequired
  ) in:
    // Verifies arc42 §10 reproducibility claims.
    // Requires GPU. To run: sbt "testOnly *RenderDeterminism*"
    pending
