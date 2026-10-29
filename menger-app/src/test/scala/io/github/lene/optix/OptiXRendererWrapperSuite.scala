package io.github.lene.optix

import java.util.Optional

import menger.common.ImageSize
import org.scalamock.scalatest.MockFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class OptiXRendererWrapperSuite extends AnyFlatSpec with Matchers with MockFactory:

  /** Wrapper whose renderer is injected, bypassing native initialization. */
  private def wrapperWith(injected: OptiXRenderer): OptiXRendererWrapper =
    new OptiXRendererWrapper:
      override protected def initializeRenderer: OptiXRenderer = injected

  private def renderResult(image: Array[Byte]): RenderResult =
    RenderResult(image, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0, 0, 0f)

  private val dims = ImageSize(4, 4)

  "renderScene" should "return the rendered bytes on success" in:
    val renderer = mock[OptiXRenderer]
    val bytes = Array[Byte](1, 2, 3)
    (renderer.render(_: ImageSize)).expects(dims).returning(bytes).once()
    wrapperWith(renderer).renderScene(dims) shouldBe Some(bytes)

  it should "return None when the renderer yields null" in:
    val renderer = mock[OptiXRenderer]
    (renderer.render(_: ImageSize)).expects(*).returning(null).once() // scalafix:ok DisableSyntax.null
    wrapperWith(renderer).renderScene(dims) shouldBe None

  it should "return None when the renderer throws (native render failure)" in:
    val renderer = mock[OptiXRenderer]
    (renderer.render(_: ImageSize)).expects(*).throwing(new RuntimeException("native render failed")).once()
    wrapperWith(renderer).renderScene(dims) shouldBe None

  // A sticky CUDA error leaves the context permanently unusable: retrying every frame (the
  // None path) only repeats the same failure forever, so it must surface as a fatal error.
  // optix-jni 0.5.0 throws OptiXContextLostException for every one (optix-jni#73), so the wrapper
  // matches the type; it parsed the error codes out of the message before.
  private def cudaFailure(description: String, code: Int): OptiXException =
    OptiXException(s"CUDA call 'cudaDeviceSynchronize()' failed: $description ($code)")

  it should "fail fast on a lost CUDA context" in:
    val renderer = mock[OptiXRenderer]
    val cause = new OptiXContextLostException(
      "CUDA context lost earlier in this process (restart it): illegal memory access (700)")
    (renderer.render(_: ImageSize)).expects(*).throwing(cause).once()
    val thrown = the[RuntimeException] thrownBy wrapperWith(renderer).renderScene(dims)
    thrown.getMessage should include("restart required")
    thrown.getMessage should include("(700)")

  it should "return None for any other OptiX failure, whatever its message says" in:
    val renderer = mock[OptiXRenderer]
    (renderer.render(_: ImageSize)).expects(*)
      .throwing(cudaFailure("unspecified launch failure", 719)).once()
    wrapperWith(renderer).renderScene(dims) shouldBe None

  it should "still return None on a recoverable CUDA error (2 out of memory)" in:
    val renderer = mock[OptiXRenderer]
    (renderer.render(_: ImageSize)).expects(*).throwing(cudaFailure("out of memory", 2)).once()
    wrapperWith(renderer).renderScene(dims) shouldBe None

  "renderSceneWithStats" should "convert a present Optional to Some" in:
    val renderer = mock[OptiXRenderer]
    val result = renderResult(Array[Byte](9))
    (renderer.renderWithStats(_: ImageSize)).expects(dims).returning(Optional.of(result)).once()
    wrapperWith(renderer).renderSceneWithStats(dims) shouldBe Some(result)

  it should "convert an empty Optional to None" in:
    val renderer = mock[OptiXRenderer]
    (renderer.renderWithStats(_: ImageSize)).expects(*).returning(Optional.empty()).once()
    wrapperWith(renderer).renderSceneWithStats(dims) shouldBe None

  "dispose" should "do nothing when no renderer has been created" in:
    val renderer = mock[OptiXRenderer] // no expectations: must never be touched
    noException should be thrownBy wrapperWith(renderer).dispose()

  it should "dispose the renderer once it has been created" in:
    val renderer = mock[OptiXRenderer]
    (renderer.dispose _).expects().once()
    val wrapper = wrapperWith(renderer)
    wrapper.renderer // force lazy creation so the renderer is cached
    wrapper.dispose()
