package menger.objects.higher_d

import menger.objects.FixedVector
import menger.objects.Vector

class RectIndices(val i0: Int, val i1: Int, val i2: Int, val i3: Int) extends FixedVector[4, Int](i0, i1, i2, i3):
  def toFace4D(vertices: Seq[Vector[4]]): Face4D =
    Face4D(values.toIndexedSeq.map(vertices.apply))

object RectIndices:
  def fromTuples(indices: (Int, Int, Int, Int)*): Seq[RectIndices] =
    indices.map { case (a, b, c, d) => RectIndices(a, b, c, d) }
