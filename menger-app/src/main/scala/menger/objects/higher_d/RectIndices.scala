package menger.objects.higher_d

import menger.common.Vector

class RectIndices(val i0: Int, val i1: Int, val i2: Int, val i3: Int)
    extends FaceIndices[4](IndexedSeq(i0, i1, i2, i3)):

  override def toFace4D(vertices: Seq[Vector[4]]): Face4D[4] =
    Face4D.fromSeq(IndexedSeq(i0, i1, i2, i3).map(vertices.apply))

object RectIndices:
  def fromTuples(indices: (Int, Int, Int, Int)*): Seq[RectIndices] =
    indices.map { case (a, b, c, d) => RectIndices(a, b, c, d) }
