package menger.objects

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.{Material, Model, ModelInstance}

case class Square(
  material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(material, primitiveType):

  def at(x: Float, y: Float, z: Float, scale: Float): List[ModelInstance] =
    at(x, y, z, scale, 1, 0, 0, 0)

  def at(
          x: Float, y: Float, z: Float, scale: Float,
          axisX: Float, axisY: Float, axisZ: Float, angle: Float
        ): List[ModelInstance] =
    val instance = new ModelInstance(model)
    instance.transform.setToTranslationAndScaling(x, y, z, scale, scale, scale)
    if angle != 0 then instance.transform.rotate(axisX, axisY, axisZ, angle)
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

