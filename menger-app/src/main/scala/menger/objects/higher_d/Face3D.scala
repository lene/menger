package menger.objects.higher_d

import menger.common.Vector

case class Face3D[V <: Int & Singleton](vertices: IndexedSeq[Vector[3]])(using v: ValueOf[V]):

  val vertsPerFace: Int = v.value
  require(vertices.size == vertsPerFace,
    s"Face3D[$vertsPerFace] requires $vertsPerFace vertices, got ${vertices.size}")

  def apply(i: Int): Vector[3] = vertices(i)
