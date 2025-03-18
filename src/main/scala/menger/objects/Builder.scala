package menger.objects

import com.badlogic.gdx.graphics.VertexAttributes.Usage
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.attributes.{ColorAttribute, IntAttribute}
import com.badlogic.gdx.graphics.g3d.utils.ModelBuilder
import com.badlogic.gdx.graphics.{Color, GL20}

object Builder:
  val modelBuilder = ModelBuilder()
  final val WHITE_MATERIAL = Material(
    ColorAttribute.createAmbient(Color(0.1, 0.1, 0.1, 1.0)),
    ColorAttribute.createDiffuse(Color(0.8, 0.8, 0.8, 1.0)),
    ColorAttribute.createSpecular(Color(1.0, 1.0, 1.0, 1.0)),
    IntAttribute.createCullFace(GL20.GL_NONE)
  )
  final val DEFAULT_FLAGS = Usage.Position | Usage.Normal
