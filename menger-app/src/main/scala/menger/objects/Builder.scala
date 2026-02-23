package menger.objects

import java.util.concurrent.atomic.AtomicReference

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.VertexAttributes.Usage
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.attributes.BlendingAttribute
import com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute
import com.badlogic.gdx.graphics.g3d.attributes.IntAttribute
import menger.gdx.ModelFactory

object Builder:
  // Injected ModelFactory - use default (LibGDX) for production, mock for tests
  private val _modelFactory: AtomicReference[ModelFactory] =
    new AtomicReference(ModelFactory.default)
  def modelFactory: ModelFactory = _modelFactory.get()
  def setModelFactory(factory: ModelFactory): Unit = _modelFactory.set(factory)

  final val WHITE_MATERIAL = material(Color(0.8, 0.8, 0.8, 1.0))
  final val DEFAULT_FLAGS = Usage.Position | Usage.Normal

  private def hasTransparency(colors: Color*): Boolean =
    colors.exists(_.a < 1.0f)

  def material(ambientColor: Color, diffuseColor: Color, specularColor: Color): Material =
    val mat = Material(
      ColorAttribute.createAmbient(ambientColor),
      ColorAttribute.createDiffuse(diffuseColor),
      ColorAttribute.createSpecular(specularColor),
      IntAttribute.createCullFace(GL20.GL_NONE)
    )
    // Enable blending if any color has alpha < 1.0
    if hasTransparency(ambientColor, diffuseColor, specularColor) then
      mat.set(new BlendingAttribute(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA))
    mat

  def material(color: Color): Material =
    material(
      color.cpy().mul(Color(0.1f, 0.1f, 0.1f, 1f)),
      color.cpy().mul(Color(0.8f, 0.8f, 0.8f, 1f)), color)