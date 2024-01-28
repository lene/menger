package menger

import com.badlogic.gdx.graphics.g2d.SpriteBatch
import com.badlogic.gdx.graphics.{GL20, Texture}
import com.badlogic.gdx.math.MathUtils
import com.badlogic.gdx.{Game, Gdx}

class EngineTest1 extends Game:

  private lazy val gdxResources = GDXResources(1)
  private lazy val builder = GeometryBuilder()
  
  override def create(): Unit = {}

  override def render(): Unit =
    gdxResources.render(builder.createModel("WHITE", 0, 0, 0, 1) :: Nil )
  
  override def dispose(): Unit =
      gdxResources.dispose()
      builder.dispose()

  override def resume(): Unit = {}

  override def resize(width: Int, height: Int): Unit = gdxResources.resize()

  override def pause(): Unit = {}
