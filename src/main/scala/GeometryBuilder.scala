package menger

import com.badlogic.gdx.graphics.{Color, GL20}
import com.badlogic.gdx.graphics.VertexAttributes.Usage
import com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute
import com.badlogic.gdx.graphics.g3d.utils.ModelBuilder
import com.badlogic.gdx.graphics.g3d.{Material, Model, ModelInstance}


object Builder:
  val modelBuilder = new ModelBuilder
  val createSphere: (Float, Float, Float, Int, Int, Material, Long) => Model = modelBuilder.createSphere
  final val WHITE_MATERIAL = Material(
    ColorAttribute.createAmbient(Color(0.1, 0.1, 0.1, 1.0)),
    ColorAttribute.createDiffuse(Color(0.8, 0.8, 0.8, 1.0)),
    ColorAttribute.createSpecular(Color(1.0, 1.0, 1.0, 1.0))
  )
  final val DEFAULT_FLAGS = Usage.Position | Usage.Normal


val SPHERE_DIVISIONS = 32
class Sphere(divisions: Int = SPHERE_DIVISIONS, material: Material = Builder.WHITE_MATERIAL):

  lazy val model: Model =
    Builder.createSphere(1, 1, 1, 2 * divisions, divisions, material, Builder.DEFAULT_FLAGS)

  def at(x: Float, y: Float, z: Float, scale: Float): List[ModelInstance] =
    val instance = new ModelInstance(model)
    instance.transform.setToTranslationAndScaling(
      x, y, z, scale, scale, scale
    )
    instance :: Nil

class Square(material: Material = Builder.WHITE_MATERIAL):
  lazy val model: Model =
    Builder.modelBuilder.createRect(
      -0.5f, -0.5f, 0,
      0.5f, -0.5f, 0,
      0.5f, 0.5f, 0,
      -0.5f, 0.5f, 0,
      0, 0, 1,
      GL20.GL_LINES,
//      GL20.GL_TRIANGLES,
      material,
//      Builder.DEFAULT_FLAGS
      Usage.Position
    )

  def at(
    x: Float, y: Float, z: Float, scale: Float,
    axisX: Float = 1, axisY: Float = 0, axisZ: Float = 0, angle: Float = 0
  ): List[ModelInstance] =
    val instance = new ModelInstance(model)
    instance.transform.setToTranslationAndScaling(
      x, y, z, scale, scale, scale
    )
    if angle != 0 then instance.transform.rotate(axisX, axisY, axisZ, angle)
    instance :: Nil

class Cube(material: Material = Builder.WHITE_MATERIAL):

  def at(x: Float, y: Float, z: Float, scale: Float): List[ModelInstance] =
    val face = Square(material)
    face.at(x, y - 0.5f * scale, z, scale, 1, 0, 0, 90) :::
    face.at(x, y + 0.5f * scale, z, scale, 1, 0, 0, -90) :::
    face.at(x - 0.5f * scale, y, z, scale, 0, 1, 0, 90) :::
    face.at(x + 0.5f * scale, y, z, scale, 0, 1, 0, -90) :::
    face.at(x, y, z - 0.5f * scale, scale, 0, 0, 1, 90) :::
    face.at(x, y, z + 0.5f * scale, scale, 0, 0, 1, -90)

class GeometryBuilder:

  val models: Map[String, Model] = createModels()
  val CENTER = 0.1f

  def dispose(): Unit = models.values.foreach(_.dispose())

  def createModels(): Map[String, Model] =
    Map("WHITE" -> Sphere().model)

  def createModel(name: String, x: Float, y: Float, z: Float, scale: Float): ModelInstance =
    val instance = new ModelInstance(models(name))
    instance.transform.setToTranslationAndScaling(
      x, y, z, scale, scale, scale
    )
    instance.transform.translate(-CENTER, -CENTER, -CENTER)
    instance
