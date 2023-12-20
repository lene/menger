package menger

import com.badlogic.gdx.backends.lwjgl3.{
  Lwjgl3Application, Lwjgl3ApplicationConfiguration
}

import com.badlogic.gdx.Game
class EngineTest1 extends Game:
  import com.badlogic.gdx.graphics.g2d.SpriteBatch
  import com.badlogic.gdx.graphics.{GL20, Texture}
  import com.badlogic.gdx.math.MathUtils
  import com.badlogic.gdx.{Game, Gdx}

  lazy val sprite = new Texture("alice.png")
  lazy val batch = new SpriteBatch

  override def create(): Unit = {}

  override def render(): Unit = {
    Gdx.gl.glClearColor(0.4f + MathUtils.random() * 0.2f, 0.4f + MathUtils.random() * 0.2f, 0.4f + MathUtils.random() * 0.2f, 1f)
    Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT)
    batch.begin()
    batch.draw(sprite, (Gdx.graphics.getWidth - sprite.getWidth) / 2f, (Gdx.graphics.getHeight - sprite.getHeight) / 2f)
    batch.end()
  }

object Main extends App:
  private final val COLOR_BITS = 8
  private final val DEPTH_BITS = 16
  private final val STENCIL_BITS = 0
  private final val NUM_ANTIALIAS_SAMPLES = 4

  val config = new Lwjgl3ApplicationConfiguration
  config.disableAudio(true)
  config.setTitle("Engine Test 1")
  config.setWindowedMode(800, 600)
  config.setBackBufferConfig(
    COLOR_BITS, COLOR_BITS, COLOR_BITS, COLOR_BITS, DEPTH_BITS, STENCIL_BITS,
    NUM_ANTIALIAS_SAMPLES
  )
  new Lwjgl3Application(new EngineTest1, config)
