package menger.objects

import scala.math.abs

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3


class SpongeByVolume(
  level: Int, material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Cube(material, primitiveType):
  private val subSponge = if level > 1
  then SpongeByVolume(level - 1, material, primitiveType)
  else Cube(material, primitiveType)

  override def at(center: Vector3, scale: Float): List[ModelInstance] =
    if level <= 0 then super.at(center, scale)
    else
      logTime("at", 10) {
        val shift = scale / 3f
        val subCubeList = for (
          xx <- -1 to 1; yy <- -1 to 1; zz <- -1 to 1 if abs(xx) + abs(yy) + abs(zz) > 1
        ) 
          yield subSponge.at(
            Vector3(center.x + xx * shift, center.y + yy * shift, center.z + zz * shift), scale / 3f
          )
        
        subCubeList.flatten.toList
      }

  override def toString: String =
    s"SpongeByVolume(level=$level, ${6 * Math.pow(20, level).toLong} faces)"
