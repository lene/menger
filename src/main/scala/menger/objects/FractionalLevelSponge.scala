package menger.objects

import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.math.Vector3


trait FractionalLevelSponge extends FractionalLevelObject:
  def center: Vector3
  def scale: Float
  def primitiveType: Int

  
  protected def createInstance(
    center: Vector3, scale: Float, level: Float, material: Material, primitiveType: Int
  ): Geometry & FractionalLevelSponge

  private[objects] lazy val transparentSponge: Option[Geometry & FractionalLevelSponge] =
    if level.isValidInt then None
    else Some(createInstance(center, scale, level.floor, transparentMaterial, primitiveType))

  private[objects] lazy val nextLevelSponge: Option[Geometry & FractionalLevelSponge] =
    if level.isValidInt then None
    else Some(createInstance(center, scale, (level+1).floor, material, primitiveType))

