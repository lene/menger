package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3

case class Face3D[V <: Int & Singleton](vertices: IndexedSeq[Vector3])(using v: ValueOf[V]):

  val vertsPerFace: Int = v.value
  require(vertices.size == vertsPerFace,
    s"Face3D[$vertsPerFace] requires $vertsPerFace vertices, got ${vertices.size}")

  def apply(i: Int): Vector3 = vertices(i)
