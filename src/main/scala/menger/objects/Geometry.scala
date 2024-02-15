package menger.objects

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.{Material, ModelInstance}

trait Geometry(material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES):
  def at(x: Float, y: Float, z: Float, scale: Float = 1): List[ModelInstance]
  override def toString: String = getClass.getSimpleName
  def logTime[T](msg: String, minDuration: Int = 0)(f: => T): T =
    val start = System.currentTimeMillis()
    val result = f
    val duration = System.currentTimeMillis() - start
    if duration >= minDuration then
    Gdx.app.log(s"${getClass.getSimpleName}.$msg", s"${duration}ms")
    result