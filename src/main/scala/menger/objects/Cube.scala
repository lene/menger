package menger.objects

import scala.collection.mutable

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.Model
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3

case class Cube(
  center: Vector3 = Vector3.Zero, scale: Float = 1f,
  material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(center, scale):

  def getModel: List[ModelInstance] =
    val instance = new ModelInstance(Cube.model(material, primitiveType))
    instance.transform.setToTranslationAndScaling(center, Vector3(scale, scale, scale))
    instance :: Nil


object Cube:
  private type CubeDefinition = (material: Material, primitiveType: Int)
  private val models: mutable.Map[CubeDefinition, Model] = mutable.Map.empty
  def model(material: Material, primitiveType: Int): Model =
    models.getOrElseUpdate(
      (material, primitiveType),
      Builder.modelBuilder.createBox(
        1f, 1f, 1f, primitiveType, material, Builder.DEFAULT_FLAGS
      ))
  def numStoredModels: Int = models.size


class CubeFromSquares(
  center: Vector3 = Vector3.Zero, scale: Float = 1f,
  material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(center, scale):
  def getModel: List[ModelInstance] =
    val faceModel = CubeFromSquares.face(material, primitiveType).model
    val facePositions = List(
      (Vector3(center.x, center.y - 0.5f * scale, center.z), Vector3(1, 0, 0), 90f),
      (Vector3(center.x, center.y + 0.5f * scale, center.z), Vector3(1, 0, 0), -90f),
      (Vector3(center.x - 0.5f * scale, center.y, center.z), Vector3(0, 1, 0), 90f),
      (Vector3(center.x + 0.5f * scale, center.y, center.z), Vector3(0, 1, 0), -90f),
      (Vector3(center.x, center.y, center.z - 0.5f * scale), Vector3(0, 0, 1), 90f),
      (Vector3(center.x, center.y, center.z + 0.5f * scale), Vector3(0, 0, 1), -90f)
    )

    facePositions.map { (position, axis, angle) =>
      val instance = new ModelInstance(faceModel)
      instance.transform.setToTranslationAndScaling(position, Vector3(scale, scale, scale))
      if angle != 0 then instance.transform.rotate(axis, angle)
      instance
    }

object CubeFromSquares:
  private val faces: mutable.Map[(Material, Int), Square] = mutable.Map.empty
  def face(material: Material, primitiveType: Int): Square =
    faces.getOrElseUpdate((material, primitiveType), Square(Vector3.Zero, 1f, material, primitiveType))
  def numStoredFaces: Int = faces.size
