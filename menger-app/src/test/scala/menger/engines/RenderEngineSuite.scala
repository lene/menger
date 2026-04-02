package menger.engines

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class RenderEngineSuite extends AnyFlatSpec with Matchers:

  private class TestEngine extends RenderEngine:
    override def create(): Unit  = ()
    override def render(): Unit  = ()
    override def resize(width: Int, height: Int): Unit = ()
    override def dispose(): Unit = ()
    override def pause(): Unit   = ()
    override def resume(): Unit  = ()

  "RenderEngine" should "be implementable without LibGDX" in:
    val engine = new TestEngine
    engine.create()
    engine.render()
    engine.resize(800, 600)
    engine.dispose()
    engine.pause()
    engine.resume()
    engine shouldBe a[RenderEngine]

  it should "not require extending Game" in:
    // TestEngine does not extend com.badlogic.gdx.Game — it compiles only if RenderEngine is pure
    val engine = new TestEngine
    engine shouldBe a[RenderEngine]
