package menger

import menger.objects.higher_d.{Projection, Rotation}

case class RotationProjectionParameters(
  rotXW: Float = 0, rotYW: Float = 0, rotZW: Float = 0, eyeW: Float = 2, screenW: Float = 1
):
  lazy val projection: Projection = Projection(eyeW, screenW)
  lazy val rotation: Rotation = Rotation(rotXW, rotYW, rotZW)

object RotationProjectionParameters:
  def apply(opts: MengerCLIOptions): RotationProjectionParameters =
    RotationProjectionParameters(
      opts.rotXW(), opts.rotYW(), opts.rotZW(), opts.projectionEyeW(), opts.projectionScreenW()
    )
