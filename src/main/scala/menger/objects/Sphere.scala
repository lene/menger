package menger.objects

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.{Material, Model, ModelInstance}
import com.badlogic.gdx.math.Vector3

import scala.collection.mutable

val SPHERE_DIVISIONS = 32
case class Sphere(
  divisions: Int = SPHERE_DIVISIONS, material: Material = Builder.WHITE_MATERIAL,
  primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(material, primitiveType):

  def at(center: Vector3, scale: Float): List[ModelInstance] =
    val instance = new ModelInstance(Sphere.model(divisions, material, primitiveType))
    instance.transform.setToTranslationAndScaling(
      center, Vector3(scale, scale, scale)
    )
    instance :: Nil

object Sphere:
  private type SphereDefinition = (divisions: Int, material: Material, primitiveType: Int)
  private val models: mutable.Map[SphereDefinition, Model] = mutable.Map.empty
  def model(divisions: Int, material: Material, primitiveType: Int): Model =
    models.getOrElseUpdate(
      (divisions, material, primitiveType),
      Builder.modelBuilder.createSphere(
        1, 1, 1, 2 * divisions, divisions, material, Builder.DEFAULT_FLAGS
    ))

  def numStoredModels: Int = models.size
