package menger

import scala.math.abs
import scala.collection.mutable

import com.badlogic.gdx.graphics.{Color, GL20}
import com.badlogic.gdx.graphics.VertexAttributes.Usage
import com.badlogic.gdx.graphics.g3d.attributes.{ColorAttribute, IntAttribute}
import com.badlogic.gdx.graphics.g3d.utils.ModelBuilder
import com.badlogic.gdx.graphics.g3d.{Material, Model, ModelInstance}

import objects.Cube

object Builder:
  val modelBuilder = new ModelBuilder
  final val WHITE_MATERIAL = Material(
    ColorAttribute.createAmbient(Color(0.1, 0.1, 0.1, 1.0)),
    ColorAttribute.createDiffuse(Color(0.8, 0.8, 0.8, 1.0)),
    ColorAttribute.createSpecular(Color(1.0, 1.0, 1.0, 1.0)),
    IntAttribute.createCullFace(GL20.GL_NONE)
  )
  final val DEFAULT_FLAGS = Usage.Position | Usage.Normal


class Square(material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES):

  def at(
    x: Float, y: Float, z: Float, scale: Float,
    axisX: Float = 1, axisY: Float = 0, axisZ: Float = 0, angle: Float = 0
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


class SpongeByVolume(
  level: Int, material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Cube(material, primitiveType):
  private val subSponge = if level > 1
  then SpongeByVolume(level - 1, material, primitiveType)
  else Cube(material, primitiveType)

  override def at(x: Float, y: Float, z: Float, scale: Float): List[ModelInstance] =
    if level <= 0 then super.at(x, y, z, scale)
    else
      val shift = scale / 3f
      val subCubeList = for (
        xx <- -1 to 1; yy <- -1 to 1; zz <- -1 to 1 if abs(xx) + abs(yy) + abs(zz) > 1
      ) yield subSponge.at(x + xx * shift, y + yy * shift, z + zz * shift, scale / 3f)
      subCubeList.flatten.toList

//object SpongeByVolume:
//  private val subSponges: mutable.Map[(Int, Material, Int), Cube] = mutable.Map.empty
//  def subSponge(level: Int, material: Material, primitiveType: Int): Cube =
//    subSponges.getOrElseUpdate(
//      (level, material, primitiveType),
//      if level > 1 then SpongeByVolume(level - 1, material, primitiveType)
//      else Cube(material, primitiveType)
//    )