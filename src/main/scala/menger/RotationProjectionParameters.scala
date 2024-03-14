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
    val newEyeW = math.pow(eyeW, if other.eyeW >= 1 then 1.1 else 0.9).toFloat
    RotationProjectionParameters(
      rotXW + other.rotXW, rotYW + other.rotYW, rotZW + other.rotZW, newEyeW, screenW
    )

object RotationProjectionParameters:
  def apply(opts: MengerCLIOptions): RotationProjectionParameters =
    RotationProjectionParameters(
      opts.rotXW(), opts.rotYW(), opts.rotZW(), opts.projectionEyeW(), opts.projectionScreenW()
    )
