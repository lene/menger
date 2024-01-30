package menger.objects

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.{Material, Model, ModelInstance}
import menger.Builder

import scala.collection.mutable

val SPHERE_DIVISIONS = 32
case class Sphere(
  divisions: Int = SPHERE_DIVISIONS, material: Material = Builder.WHITE_MATERIAL, 
  primitiveType: Int = GL20.GL_TRIANGLES
):

  def at(x: Float, y: Float, z: Float, scale: Float): List[ModelInstance] =
    val instance = new ModelInstance(Sphere.model(divisions, material, primitiveType))
    instance.transform.setToTranslationAndScaling(
      x, y, z, scale, scale, scale
    )
    instance :: Nil

object Sphere:
  private val models: mutable.Map[(Int, Material, Int), Model] = mutable.Map.empty
  def model(divisions: Int, material: Material, primitiveType: Int): Model =
    models.getOrElseUpdate(
      (divisions, material, primitiveType),
      Builder.modelBuilder.createSphere(
        1, 1, 1, 2 * divisions, divisions, material, Builder.DEFAULT_FLAGS
    ))
    
  def numStoredModels: Int = models.size