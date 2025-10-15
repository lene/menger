package menger.objects.higher_d

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import menger.RotationProjectionParameters
import menger.objects.Builder
import menger.objects.FractionalLevelObject
import menger.objects.Geometry
import menger.objects.higher_d.Mesh4D
import menger.objects.higher_d.Projection
import menger.objects.higher_d.RotatedProjection
import menger.objects.higher_d.Rotation

/** Wraps RotatedProjection to support fractional levels for 4D sponges */
class FractionalRotatedProjection(
  center: Vector3 = Vector3.Zero, scale: Float = 1f,
  object4DFactory: Float => Mesh4D, val level: Float,
  @SuppressWarnings(Array("org.wartremover.warts.Var")) var projection: Projection,
  @SuppressWarnings(Array("org.wartremover.warts.Var")) var rotation: Rotation = Rotation(),
  val material: Material = Builder.WHITE_MATERIAL,
  primitiveType: Int = GL20.GL_TRIANGLES
)(using val profilingConfig: menger.ProfilingConfig) extends Geometry(center, scale) with FractionalLevelObject:

  private lazy val transparentRotatedProjection: Option[RotatedProjection] =
    if level.isValidInt then None
    else Some(RotatedProjection(
      center, scale, object4DFactory(level.floor), projection, rotation,
      materialWithAlpha(calculateAlpha()), primitiveType
    ))

  private lazy val nextLevelRotatedProjection: Option[RotatedProjection] =
    if level.isValidInt then None
    else Some(RotatedProjection(
      center, scale, object4DFactory((level + 1).floor), projection, rotation, material, primitiveType
    ))

  private lazy val integerRotatedProjection: Option[RotatedProjection] =
    if level.isValidInt then Some(RotatedProjection(center, scale, object4DFactory(level), projection, rotation, material, primitiveType))
    else None

  override def getModel: List[ModelInstance] =
    if level.isValidInt then integerRotatedProjection.map(_.getModel).getOrElse(Nil)
    else List(
      nextLevelRotatedProjection.map(_.getModel).getOrElse(Nil),
      transparentRotatedProjection.map(_.getModel).getOrElse(Nil)
    ).flatten

  override def handleEvent(event: RotationProjectionParameters): Unit =
    rotation += event.rotation
    projection += event.projection
    transparentRotatedProjection.foreach(_.handleEvent(event))
    nextLevelRotatedProjection.foreach(_.handleEvent(event))
    integerRotatedProjection.foreach(_.handleEvent(event))

  override def toString: String = s"FractionalRotatedProjection[${object4DFactory(level)}]"
