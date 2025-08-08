package menger.objects.higher_d

import menger.objects.Vector

class RectIndices(i0: Int, i1: Int, i2: Int, i3: Int) extends FixedVector[4, Int](i0, i1, i2, i3):
  def toFace4D(vertices: Seq[Vector[4, Float]]): Face4D =
    Face4D(values.map(vertices.apply))

object RectIndices:
  def fromTuples(indices: (Int, Int, Int, Int)*): Seq[RectIndices] =
    indices.map { case (a, b, c, d) => RectIndices(a, b, c, d) }
