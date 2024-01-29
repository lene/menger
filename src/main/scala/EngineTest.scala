package menger

import com.badlogic.gdx.graphics.g2d.SpriteBatch
import com.badlogic.gdx.graphics.{GL20, Texture}
import com.badlogic.gdx.math.MathUtils
import com.badlogic.gdx.{Game, Gdx}

class EngineTest extends Game:

  private lazy val gdxResources = GDXResources(1)
  private lazy val builder = GeometryBuilder()
  
  override def create(): Unit = {}

  override def render(): Unit =
    gdxResources.render(/*Sphere().at(0, 0, 0, 1) ::*/ Cube().at(0, 0, 0, 2) )

  override def dispose(): Unit =
      gdxResources.dispose()
      builder.dispose()

  override def resume(): Unit = {}

  override def resize(width: Int, height: Int): Unit = gdxResources.resize()

  override def pause(): Unit = {}
