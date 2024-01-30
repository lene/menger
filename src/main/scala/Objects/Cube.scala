package menger.objects

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.{Material, Model, ModelInstance}
import menger.{Builder, Square}

import scala.collection.mutable

case class Cube(
  material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
):

  def at(x: Float, y: Float, z: Float, scale: Float): List[ModelInstance] =
    val instance = new ModelInstance(Cube.model(material, primitiveType))
    instance.transform.setToTranslationAndScaling(
      x, y, z, scale, scale, scale
    )
    instance :: Nil

  def old_at(x: Float, y: Float, z: Float, scale: Float): List[ModelInstance] =
    val face = Cube.face(material, primitiveType)
    face.at(x, y - 0.5f * scale, z, scale, 1, 0, 0, 90) :::
      face.at(x, y + 0.5f * scale, z, scale, 1, 0, 0, -90) :::
      face.at(x - 0.5f * scale, y, z, scale, 0, 1, 0, 90) :::
      face.at(x + 0.5f * scale, y, z, scale, 0, 1, 0, -90) :::
      face.at(x, y, z - 0.5f * scale, scale, 0, 0, 1, 90) :::
      face.at(x, y, z + 0.5f * scale, scale, 0, 0, 1, -90)


object Cube:
  private val models: mutable.Map[(Material, Int), Model] = mutable.Map.empty
  def model(material: Material, primitiveType: Int): Model =
    models.getOrElseUpdate(
      (material, primitiveType),
      Builder.modelBuilder.createBox(
        1f, 1f, 1f, primitiveType, material, Builder.DEFAULT_FLAGS
      ))
  def numStoredModels: Int = models.size

  private val faces: mutable.Map[(Material, Int), Square] = mutable.Map.empty
  def face(material: Material, primitiveType: Int): Square =
    faces.getOrElseUpdate((material, primitiveType), Square(material, primitiveType))
  def numStoredFaces: Int = faces.size
