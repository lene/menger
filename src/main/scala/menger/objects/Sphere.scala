package menger.objects

import java.util.concurrent.ConcurrentHashMap

import scala.collection.mutable
import scala.jdk.CollectionConverters._

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.Model
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3

val SPHERE_DIVISIONS = 32
case class Sphere(
  center: Vector3 = Vector3.Zero, scale: Float = 1f,
  divisions: Int = SPHERE_DIVISIONS, material: Material = Builder.WHITE_MATERIAL,
  primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(center, scale):

  def getModel: List[ModelInstance] =
    val instance = new ModelInstance(Sphere.model(divisions, material, primitiveType))
    instance.transform.setToTranslationAndScaling(
      center, Vector3(scale, scale, scale)
    )
    instance :: Nil

object Sphere:
  private type SphereDefinition = (divisions: Int, material: Material, primitiveType: Int)
  private val models: mutable.Map[SphereDefinition, Model] =
    new ConcurrentHashMap[SphereDefinition, Model]().asScala
  def model(divisions: Int, material: Material, primitiveType: Int): Model =
    models.getOrElseUpdate(
      (divisions, material, primitiveType),
      Builder.modelBuilder.createSphere(
        1, 1, 1, 2 * divisions, divisions, material, Builder.DEFAULT_FLAGS
    ))

  def numStoredModels: Int = models.size
