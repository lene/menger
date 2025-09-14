package menger.objects

import com.badlogic.gdx.graphics.VertexAttributes.Usage
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.attributes.{ColorAttribute, IntAttribute}
import com.badlogic.gdx.graphics.g3d.utils.ModelBuilder
import com.badlogic.gdx.graphics.{Color, GL20}

object Builder:
  val modelBuilder = ModelBuilder()
  final val WHITE_MATERIAL = material(Color(0.8, 0.8, 0.8, 1.0))
  final val DEFAULT_FLAGS = Usage.Position | Usage.Normal

  def material(ambientColor: Color, diffuseColor: Color, specularColor: Color): Material =
    Material(
      ColorAttribute.createAmbient(ambientColor),
      ColorAttribute.createDiffuse(diffuseColor),
      ColorAttribute.createSpecular(specularColor),
      IntAttribute.createCullFace(GL20.GL_NONE)
    )

  def material(color: Color): Material =
    material(
      color.cpy().mul(Color(0.1f, 0.1f, 0.1f, 1f)),
      color.cpy().mul(Color(0.8f, 0.8f, 0.8f, 1f)), color)