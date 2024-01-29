package menger

import com.badlogic.gdx.graphics.g2d.SpriteBatch
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.graphics.{GL20, Texture}
import com.badlogic.gdx.math.MathUtils
import com.badlogic.gdx.utils.Timer
import com.badlogic.gdx.{Game, Gdx}

class EngineTest(timeout: Float = 0, spongeLevel: Int = 0, lines: Boolean = false) extends Game:

  private lazy val gdxResources = GDXResources()
  private lazy val primitiveType = if lines then GL20.GL_LINES else GL20.GL_TRIANGLES
  private lazy val sponge: List[ModelInstance] =
    SpongeByVolume(spongeLevel, primitiveType = primitiveType).at(0, 0, 0, 1)
  
  override def create(): Unit =
    Gdx.app.log("create", s"Level $spongeLevel, ${sponge.size} faces")
    if timeout > 0 then
      Timer.schedule(() => Gdx.app.exit(), timeout, 0)

  override def render(): Unit = gdxResources.render(sponge)

  override def dispose(): Unit = gdxResources.dispose()

  override def resume(): Unit = {}

  override def resize(width: Int, height: Int): Unit = gdxResources.resize()

  override def pause(): Unit = {}
