package menger.objects

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.{Material, ModelInstance}

trait Geometry(material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES):
  def at(x: Float, y: Float, z: Float, scale: Float = 1): List[ModelInstance]
