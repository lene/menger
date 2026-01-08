package menger.objects.higher_d

import menger.common.float2string

trait Fractal4D(val level: Float) extends Mesh4D:
  override def toString: String = s"${getClass.getSimpleName}(level=${float2string(level)}, ${faces.size} faces)"

