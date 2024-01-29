package menger

import scala.math.abs

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.{Color, GL20}
import com.badlogic.gdx.graphics.VertexAttributes.Usage
import com.badlogic.gdx.graphics.g3d.attributes.{ColorAttribute, IntAttribute}
import com.badlogic.gdx.graphics.g3d.utils.ModelBuilder
import com.badlogic.gdx.graphics.g3d.{Material, Model, ModelInstance}


object Builder:
  val modelBuilder = new ModelBuilder
  final val WHITE_MATERIAL = Material(
    ColorAttribute.createAmbient(Color(0.1, 0.1, 0.1, 1.0)),
    ColorAttribute.createDiffuse(Color(0.8, 0.8, 0.8, 1.0)),
    ColorAttribute.createSpecular(Color(1.0, 1.0, 1.0, 1.0)),
    IntAttribute.createCullFace(GL20.GL_NONE)
  )
  final val DEFAULT_FLAGS = Usage.Position | Usage.Normal


val SPHERE_DIVISIONS = 32
class Sphere(divisions: Int = SPHERE_DIVISIONS, material: Material = Builder.WHITE_MATERIAL):

  lazy val model: Model =
    Builder.modelBuilder.createSphere(1, 1, 1, 2 * divisions, divisions, material, Builder.DEFAULT_FLAGS)

  def at(x: Float, y: Float, z: Float, scale: Float): List[ModelInstance] =
    val instance = new ModelInstance(model)
    instance.transform.setToTranslationAndScaling(
      x, y, z, scale, scale, scale
    )
    instance :: Nil

class Square(material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES):
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

class Cube(material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES):

  def at(x: Float, y: Float, z: Float, scale: Float): List[ModelInstance] =
    val face = Square(material, primitiveType)
    face.at(x, y - 0.5f * scale, z, scale, 1, 0, 0, 90) :::
    face.at(x, y + 0.5f * scale, z, scale, 1, 0, 0, -90) :::
    face.at(x - 0.5f * scale, y, z, scale, 0, 1, 0, 90) :::
    face.at(x + 0.5f * scale, y, z, scale, 0, 1, 0, -90) :::
    face.at(x, y, z - 0.5f * scale, scale, 0, 0, 1, 90) :::
    face.at(x, y, z + 0.5f * scale, scale, 0, 0, 1, -90)

class SpongeByVolume(
  level: Int, material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Cube(material, primitiveType):
  override def at(x: Float, y: Float, z: Float, scale: Float): List[ModelInstance] =
    if level <= 0 then super.at(x, y, z, scale)
    else
      val subSponge = SpongeByVolume(level - 1, material, primitiveType)
      val shift = scale / 3f
      val subCubeList = for (
        xx <- -1 to 1; yy <- -1 to 1; zz <- -1 to 1 if abs(xx) + abs(yy) + abs(zz) > 1
      ) yield subSponge.at(x + xx * shift, y + yy * shift, z + zz * shift, scale / 3f)
      subCubeList.flatten.toList
