package menger.objects

import scala.math.abs

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import menger.common.float2string


class SpongeByVolume(
  override val center: Vector3 = Vector3.Zero, override val scale: Float = 1f,
  val level: Float, override val material: Material = Builder.WHITE_MATERIAL, override val primitiveType: Int = GL20.GL_TRIANGLES
) extends Cube(center, scale, material, primitiveType) with FractionalLevelSponge:

  override protected def createInstance(
    center: Vector3, scale: Float, level: Float, material: Material, primitiveType: Int
  ): Geometry & FractionalLevelSponge =
    SpongeByVolume(center, scale, level, material, primitiveType)

  private val subSponge = if level > 1
  then SpongeByVolume(Vector3.Zero, 1f, level - 1, material, primitiveType)
  else Cube(Vector3.Zero, 1f, material, primitiveType)

  override def getModel: List[ModelInstance] =
    if level <= 0 then super.getModel
    else if level.isValidInt then getIntegerModel
    else List(
      nextLevelSponge.map(_.getModel).getOrElse(Nil),
      transparentSponge.map(_.getModel).getOrElse(Nil)
    ).flatten

  private lazy val getIntegerModel =
    val shift = scale / 3f
    val subCubeList = for (
      xx <- -1 to 1; yy <- -1 to 1; zz <- -1 to 1 if abs(xx) + abs(yy) + abs(zz) > 1
    )
    yield SpongeByVolume(
      Vector3(center.x + xx * shift, center.y + yy * shift, center.z + zz * shift), scale / 3f,
      level - 1, material, primitiveType
    ).getModel

    subCubeList.flatten.toList

  override def toString: String =
    s"SpongeByVolume(level=${float2string(level)}, ${6 * Math.pow(20, level.toInt).toLong} faces)"
