package menger.objects.higher_d

trait Fractal4D(val level: Int) extends Mesh4D:
  override def toString: String = s"${getClass.getSimpleName}(level=$level, ${faces.size} faces)"

