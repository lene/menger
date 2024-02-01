package menger

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.{Material, ModelInstance}

import scala.math.abs


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