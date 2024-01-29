package menger

import com.badlogic.gdx.graphics.g2d.SpriteBatch
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.graphics.{GL20, Texture}
import com.badlogic.gdx.math.MathUtils
import com.badlogic.gdx.utils.Timer
import com.badlogic.gdx.{Game, Gdx}

class EngineTest(timeout: Float = 0, spongeLevel: Int = 0) extends Game:

  private lazy val gdxResources = GDXResources()
  private lazy val sponge: List[ModelInstance] = SpongeByVolume(spongeLevel).at(0, 0, 0, 1)
  
  override def create(): Unit =
    println(s"${sponge.size} faces")
    if timeout > 0 then
      Timer.schedule(() => Gdx.app.exit(), timeout, 0)

  override def render(): Unit = gdxResources.render(sponge)

  override def dispose(): Unit = gdxResources.dispose()

  override def resume(): Unit = {}

  override def resize(width: Int, height: Int): Unit = gdxResources.resize()

  override def pause(): Unit = {}
