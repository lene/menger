package menger.objects

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.Model
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3

case class Square(
  center: Vector3 = Vector3.Zero, scale: Float = 1f,
  material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(center, scale):

  def getModel: List[ModelInstance] =
    at(Vector3(1, 0, 0), 0)

  def at(axis: Vector3, angle: Float): List[ModelInstance] =
    val instance = new ModelInstance(model)
    instance.transform.setToTranslationAndScaling(center, Vector3(scale, scale, scale))
    if angle != 0 then instance.transform.rotate(axis, angle)
    instance :: Nil

  lazy val model: Model =
    Builder.modelBuilder.createRect(
      -0.5f, -0.5f, 0,
      0.5f, -0.5f, 0,
      0.5f, 0.5f, 0,
      -0.5f, 0.5f, 0,
      0, 0, 1,
      primitiveType,
      material,
      Builder.DEFAULT_FLAGS
    )

