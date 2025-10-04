package menger.objects.higher_d

trait Fractal4D(val level: Float) extends Mesh4D:
  override def toString: String = s"${getClass.getSimpleName}(level=${menger.objects.float2string(level)}, ${faces.size} faces)"

