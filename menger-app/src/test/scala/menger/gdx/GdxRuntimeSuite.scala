package menger.gdx

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Tests for GdxRuntime — null-safe LibGDX wrapper.
 *
 * LibGDX is not initialised in unit tests (all Gdx.* fields are null).
 * Every method must be safe to call in that state.
 */
class GdxRuntimeSuite extends AnyFlatSpec with Matchers:

  "GdxRuntime.exit" should "not throw when Gdx.app is null" in {
    noException should be thrownBy GdxRuntime.exit()
  }

  "GdxRuntime.requestRendering" should "not throw when Gdx.graphics is null" in {
    noException should be thrownBy GdxRuntime.requestRendering()
  }

  "GdxRuntime.setContinuousRendering" should "not throw when Gdx.graphics is null" in {
    noException should be thrownBy GdxRuntime.setContinuousRendering(false)
  }

  "GdxRuntime.deltaTime" should "return 0f when Gdx.graphics is null" in {
    GdxRuntime.deltaTime shouldBe 0f
  }

  "GdxRuntime.width" should "return 0 when Gdx.graphics is null" in {
    GdxRuntime.width shouldBe 0
  }

  "GdxRuntime.height" should "return 0 when Gdx.graphics is null" in {
    GdxRuntime.height shouldBe 0
  }

  "GdxRuntime.glClear" should "not throw when Gdx.gl is null" in {
    noException should be thrownBy GdxRuntime.glClear(0x4000)
  }

  "GdxRuntime.setInputProcessor" should "not throw when Gdx.input is null" in {
    noException should be thrownBy GdxRuntime.setInputProcessor(null)
  }

  "GdxRuntime.isKeyPressed" should "return false when Gdx.input is null" in {
    GdxRuntime.isKeyPressed(0) shouldBe false
  }

  "GdxRuntime.isButtonPressed" should "return false when Gdx.input is null" in {
    GdxRuntime.isButtonPressed(0) shouldBe false
  }
