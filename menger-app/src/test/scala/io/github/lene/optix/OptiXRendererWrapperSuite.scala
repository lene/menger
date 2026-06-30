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
    wrapperWith(renderer).renderScene(dims) shouldBe bytes

  it should "return an empty array when the renderer yields null" in:
    val renderer = mock[OptiXRenderer]
    (renderer.render(_: ImageSize)).expects(*).returning(null).once() // scalafix:ok DisableSyntax.null
    wrapperWith(renderer).renderScene(dims) shouldBe Array.emptyByteArray

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
