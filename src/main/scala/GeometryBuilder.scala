package menger

import com.badlogic.gdx.graphics.{Color, GL20}
import com.badlogic.gdx.graphics.VertexAttributes.Usage
import com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute
import com.badlogic.gdx.graphics.g3d.utils.ModelBuilder
import com.badlogic.gdx.graphics.g3d.{Material, Model, ModelInstance, RenderableProvider}

import scala.annotation.tailrec

class GeometryBuilder:
  final val SPHERE_DIVISIONS = 32
  final val WHITE_MATERIAL = new Material(
    ColorAttribute.createAmbient(new Color(0.1, 0.1, 0.1, 1.0)),
    ColorAttribute.createDiffuse(new Color(0.8, 0.8, 0.8, 1.0)),
    ColorAttribute.createSpecular(new Color(1.0, 1.0, 1.0, 1.0))
  )

  val modelBuilder = new ModelBuilder
  val models: Map[String, Model] = createModels()
  val CENTER = 0.1f

  def dispose(): Unit = models.values.foreach(_.dispose())

  def createModels(): Map[String, Model] =
    Map(
      "WHITE" -> modelBuilder.createSphere(
        1, 1, 1, 2 * SPHERE_DIVISIONS, SPHERE_DIVISIONS,
        WHITE_MATERIAL, Usage.Position | Usage.Normal
      ),
    )

  def createModel(name: String, x: Float, y: Float, z: Float, scale: Float): ModelInstance =
    val instance = new ModelInstance(models(name))
    instance.transform.setToTranslationAndScaling(
      x, y, z, scale, scale, scale
    )
    instance.transform.translate(-CENTER, -CENTER, -CENTER)
    instance
