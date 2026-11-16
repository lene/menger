package menger.optix

import io.github.lene.optix.OptiXException
import io.github.lene.optix.OptiXRenderer
import menger.common.ImageSize
import org.scalamock.scalatest.MockFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// menger#179: what the wrapper tells the engines about the device's free memory and whether the
// renderer has made its first render.
class OptiXRendererWrapperMemorySuite extends AnyFlatSpec with Matchers with MockFactory:

  private def wrapperWith(injected: OptiXRenderer): OptiXRendererWrapper =
    new OptiXRendererWrapper:
      override protected def initializeRenderer: OptiXRenderer = injected

  private val dims = ImageSize(4, 4)

  "freeGpuMemoryBytes" should "be unknown, and create no renderer, before the renderer exists" in:
    // an uninitialised renderer would fail the test: `initializeRenderer` is never called
    val wrapper = new OptiXRendererWrapper:
      override protected def initializeRenderer: OptiXRenderer =
        fail("asking for the free memory must not create the renderer")
    wrapper.freeGpuMemoryBytes shouldBe None

  it should "report the free memory the renderer measures" in:
    val renderer = mock[OptiXRenderer]
    (renderer.freeGpuMemoryBytes _).expects().returning(4_000_000_000L).once()
    val wrapper = wrapperWith(renderer)
    val _ = wrapper.renderer
    wrapper.freeGpuMemoryBytes shouldBe Some(4_000_000_000L)

  it should "be unknown when the measurement fails, so the caller keeps its fixed budget" in:
    val renderer = mock[OptiXRenderer]
    (renderer.freeGpuMemoryBytes _).expects()
      .throwing(OptiXException("cudaMemGetInfo failed")).once()
    val wrapper = wrapperWith(renderer)
    val _ = wrapper.renderer
    wrapper.freeGpuMemoryBytes shouldBe None

  "hasRendered" should "become true with the first rendered frame, not before" in:
    val renderer = mock[OptiXRenderer]
    (renderer.render(_: ImageSize)).expects(dims).returning(Array[Byte](1)).once()
    val wrapper = wrapperWith(renderer)
    wrapper.hasRendered shouldBe false
    val _ = wrapper.renderScene(dims)
    wrapper.hasRendered shouldBe true

  it should "stay false when the render fails" in:
    val renderer = mock[OptiXRenderer]
    (renderer.render(_: ImageSize)).expects(*).returning(null).once() // scalafix:ok DisableSyntax.null
    val wrapper = wrapperWith(renderer)
    val _ = wrapper.renderScene(dims)
    wrapper.hasRendered shouldBe false
