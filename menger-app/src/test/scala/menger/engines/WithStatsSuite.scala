package menger.engines

import java.util.concurrent.atomic.AtomicBoolean

import io.github.lene.optix.OptiXRendererWrapper
import io.github.lene.optix.RenderResult
import menger.common.ImageSize
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Tests the JSON formatting and render-path branch of WithStats (F13).
  * No GPU needed: statsJson is pure formatting; the render-path branch uses
  * a stub that doesn't touch the native renderer. */
class WithStatsSuite extends AnyFlatSpec with Matchers:

  private class TestWithStats(
    statsEnabled: Boolean = false,
    jsonPath: Option[String] = None,
    stubWrapper: OptiXRendererWrapper = null
  ) extends WithStats:
    override protected def enableStats: Boolean = statsEnabled
    override protected def statsJsonPath: Option[String] = jsonPath
    override protected def rendererWrapper: OptiXRendererWrapper = stubWrapper

  private def sampleResult: RenderResult =
    RenderResult(
      Array.emptyByteArray, 1000000L, 500000L, 200000L, 100000L, 150000L,
      30000L, 0L, 20000L, 8, 2, 16.5f
    )

  "WithStats.statsJson" should "produce valid JSON with all fields" in:
    val tws = TestWithStats()
    tws.lastRenderResult.set(Some(sampleResult))
    val json = tws.statsJson.getOrElse(fail("expected stats JSON"))
    json should include("\"frameMs\": 16.5")
    json should include("\"totalRays\": 1000000")
    json should include("\"primaryRays\": 500000")
    json should include("\"reflectedRays\": 200000")
    json should include("\"refractedRays\": 100000")
    json should include("\"shadowRays\": 150000")
    json should include("\"aaRays\": 30000")
    json should include("\"spectralRays\": 20000")
    json should include("\"msPerMray\"")
    json.trim should startWith ("{")
    json.trim should endWith ("}")

  it should "return None when no render result exists" in:
    val tws = TestWithStats()
    tws.statsJson shouldBe None

  it should "compute msPerMray from frameMs and totalRays" in:
    val tws = TestWithStats()
    tws.lastRenderResult.set(Some(sampleResult))
    val json = tws.statsJson.getOrElse(fail())
    // msPerMray = frameMs / (totalRays / 1_000_000) = 16.5 / 1.0 = 16.5
    json should include("\"msPerMray\": 16.5")

  "WithStats.maybeRenderWithStats" should "use plain renderScene when stats disabled" in:
    val wrapper = StubWrapper(Array[Byte](1, 2, 3))
    val tws = TestWithStats(statsEnabled = false, stubWrapper = wrapper)
    val result = tws.maybeRenderWithStats(1920, 1080)
    result.isDefined shouldBe true
    wrapper.statsRenderCalled.get() shouldBe false
    wrapper.plainRenderCalled.get() shouldBe true

  it should "use renderSceneWithStats when stats enabled" in:
    val wrapper = StubWrapper(Array.emptyByteArray)
    val tws = TestWithStats(statsEnabled = true, stubWrapper = wrapper)
    tws.maybeRenderWithStats(1920, 1080)
    wrapper.statsRenderCalled.get() shouldBe true
    wrapper.plainRenderCalled.get() shouldBe false
    tws.lastRenderResult.get shouldBe defined

  private class StubWrapper(imageBytes: Array[Byte]) extends OptiXRendererWrapper(1):
    val plainRenderCalled = AtomicBoolean(false)
    val statsRenderCalled = AtomicBoolean(false)
    override def renderScene(size: ImageSize): Option[Array[Byte]] =
      plainRenderCalled.set(true)
      Some(imageBytes)
    override def renderSceneWithStats(size: ImageSize): Option[RenderResult] =
      statsRenderCalled.set(true)
      Some(sampleResult)
