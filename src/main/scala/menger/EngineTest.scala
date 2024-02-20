package menger

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import com.badlogic.gdx.utils.Timer
import com.badlogic.gdx.{Game, Gdx}
import menger.objects.{Geometry, SpongeBySurface, SpongeByVolume}

class EngineTest(
  timeout: Float = 0, spongeLevel: Int = 0, lines: Boolean = false, spongeType: String = "box"
) extends Game:

  private lazy val gdxResources = GDXResources()
  private lazy val primitiveType = if lines then GL20.GL_LINES else GL20.GL_TRIANGLES
  private lazy val sponge: Geometry =
    if spongeType == "box" then SpongeByVolume(spongeLevel, primitiveType = primitiveType)
    else SpongeBySurface(spongeLevel, primitiveType = primitiveType)
  private lazy val drawables: List[ModelInstance] =sponge.at(Vector3(0, 0, 0), 1)
  
  override def create(): Unit =
    Gdx.app.log("create", sponge.toString())
    if timeout > 0 then
      Timer.schedule(() => Gdx.app.exit(), timeout, 0)

  override def render(): Unit = gdxResources.render(drawables)

  override def dispose(): Unit = gdxResources.dispose()

  override def resume(): Unit = {}

  override def resize(width: Int, height: Int): Unit = gdxResources.resize()

  override def pause(): Unit = {}
