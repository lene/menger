package menger

import menger.objects.higher_d.{Projection, Rotation}

import scala.annotation.targetName

case class RotationProjectionParameters(
  rotXW: Float = 0, rotYW: Float = 0, rotZW: Float = 0, eyeW: Float = 2, screenW: Float = 1
):
  lazy val projection: Projection = Projection(eyeW, screenW)
  lazy val rotation: Rotation = Rotation(rotXW, rotYW, rotZW)
  
  @targetName("plus")
  def +(other: RotationProjectionParameters): RotationProjectionParameters =
    val rot = rotation + other.rotation
    val proj = projection + other.projection
    RotationProjectionParameters(rot.rotXW, rot.rotYW, rot.rotZW, proj.eyeW, proj.screenW)

object RotationProjectionParameters:
  def apply(opts: MengerCLIOptions): RotationProjectionParameters =
    RotationProjectionParameters(
      opts.rotXW(), opts.rotYW(), opts.rotZW(), opts.projectionEyeW(), opts.projectionScreenW()
    )

  def apply(rotXW: Float, rotYW: Float, rotZW: Float): RotationProjectionParameters =
    RotationProjectionParameters(rotXW, rotYW, rotZW, 2, 1)

