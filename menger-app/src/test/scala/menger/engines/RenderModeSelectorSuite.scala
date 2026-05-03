package menger.engines

import menger.ObjectSpec
import menger.ProfilingConfig
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class RenderModeSelectorSuite extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig.disabled

  private def spec(t: String) = ObjectSpec(objectType = t, x = 0, y = 0, z = 0, size = 1)

  "RenderModeSelector.classify" should "classify all-cube scene as TriangleMeshes" in:
    val specs = List(spec("cube"), spec("cube"))
    RenderModeSelector.classify(specs) shouldBe a [SceneType.TriangleMeshes]

  it should "classify parametric scene as TriangleMeshes" in:
    val specs = List(spec("parametric"))
    RenderModeSelector.classify(specs) shouldBe a [SceneType.TriangleMeshes]

  it should "classify sponge-volume scene as TriangleMeshes" in:
    val specs = List(spec("sponge-volume"))
    RenderModeSelector.classify(specs) shouldBe a [SceneType.TriangleMeshes]

  it should "classify sponge-recursive-ias scene as TriangleMeshes" in:
    val specs = List(spec("sponge-recursive-ias"))
    RenderModeSelector.classify(specs) shouldBe a [SceneType.TriangleMeshes]

  it should "classify tesseract scene as TriangleMeshes" in:
    val specs = List(spec("tesseract"))
    RenderModeSelector.classify(specs) shouldBe a [SceneType.TriangleMeshes]

  it should "classify all-sphere scene as SimpleMixed (routed by registry)" in:
    val specs = List(spec("sphere"), spec("sphere"))
    RenderModeSelector.classify(specs) shouldBe a [SceneType.SimpleMixed]

  it should "classify all-cube-sponge scene as SimpleMixed (routed by registry)" in:
    val specs = List(spec("cube-sponge"))
    RenderModeSelector.classify(specs) shouldBe a [SceneType.SimpleMixed]

  it should "classify sphere + cube as SimpleMixed" in:
    val specs = List(spec("sphere"), spec("cube"))
    RenderModeSelector.classify(specs) shouldBe a [SceneType.SimpleMixed]

  it should "classify cube-sponge + sponge-volume as SimpleMixed" in:
    val specs = List(spec("cube-sponge"), spec("sponge-volume"))
    RenderModeSelector.classify(specs) shouldBe a [SceneType.SimpleMixed]

  it should "classify sphere + cube + sponge-volume as SimpleMixed" in:
    val specs = List(spec("sphere"), spec("cube"), spec("sponge-volume"))
    RenderModeSelector.classify(specs) shouldBe a [SceneType.SimpleMixed]

  it should "classify SpongeShowcase mix as SimpleMixed" in:
    val specs = List(spec("sponge-volume"), spec("sponge-surface"), spec("cube-sponge"))
    RenderModeSelector.classify(specs) shouldBe a [SceneType.SimpleMixed]

  it should "throw IllegalArgumentException for empty spec list" in:
    an [IllegalArgumentException] should be thrownBy RenderModeSelector.classify(List.empty)
