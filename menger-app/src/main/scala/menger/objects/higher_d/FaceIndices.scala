package menger.objects.higher_d

import menger.common.Vector

class FaceIndices[V <: Int & Singleton](val indices: IndexedSeq[Int])(using v: ValueOf[V]):

  val vertsPerFace: Int = v.value
  val dimension: Int = vertsPerFace
  val values: IndexedSeq[Int] = indices
  require(indices.size == vertsPerFace,
    s"FaceIndices[$vertsPerFace] requires $vertsPerFace indices, got ${indices.size}")

  def apply(i: Int): Int =
    require(i >= 0 && i < vertsPerFace,
      s"Index must be between 0 and ${vertsPerFace - 1}, got $i")
    indices(i)

  def toFace4D(vertices: Seq[Vector[4]]): Face4D[V] =
    require(vertices.size >= vertsPerFace,
      "Not enough vertices to create Face4D")
    Face4D[V](indices.map(vertices.apply))

  def toIndexedSeq: IndexedSeq[Int] = indices

  override def equals(other: Any): Boolean = other match
    case that: FaceIndices[?] => indices == that.indices
    case _ => false

  override def hashCode(): Int = indices.hashCode()

object FaceIndices:
  def fromSeq[V <: Int & Singleton](indices: Int*)(using ValueOf[V]): FaceIndices[V] =
    new FaceIndices[V](indices.toIndexedSeq)
