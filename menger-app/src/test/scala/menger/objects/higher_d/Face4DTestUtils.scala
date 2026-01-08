package menger.objects.higher_d

import menger.common.Const
import menger.common.Vector


object Face4DTestUtils:

  def faceToString(seq: Seq[Vector[4]]): String =
    seq.map(_.toString).mkString("(", ", ", ")")

  val unitVectors: Set[Vector[4]] = Set(
    Vector.X, Vector.Y, Vector.Z, Vector.W,
    -Vector.X, -Vector.Y, -Vector.Z, -Vector.W
  )

  val positiveUnitVectors: Set[Vector[4]] =
    unitVectors.map(vec => vec * vec.filter(_ != 0).sum)

  def normals(vecs: Seq[Vector[4]]): Set[Vector[4]] =
    require(vecs.size == 2, s"Need 2 vectors, have ${vecs.size}: ${vecs.map(_.toString)}")
    require(unitVectors.contains(vecs.head), s"vec1 must be a unit vector, is ${vecs.head}")
    require(unitVectors.contains(vecs(1)), s"vec2 must be a unit vector, is ${vecs(1)}")
    require(vecs.head != vecs(1), s"vec1 and vec2 must be different, are ${vecs.head}")
    val normals = positiveUnitVectors.filter(vec => vec * vecs.head == 0 && vec * vecs(1) == 0)
    require(
      normals.size == 2,
      s"Expected 2 normals, have ${normals.size}: ${normals.map(_.toString)}"
    )
    normals

  def setIndices(v: Vector[4]): Seq[Int] =
    allIndicesWhere(v.toIndexedSeq, s => s.abs > Const.epsilon)

  def allIndicesWhere[A](s: Seq[A], pred: A => Boolean): Seq[Int] =
    s.zipWithIndex.filter { case (elem, _) => pred(elem) }.map { case (_, index) => index }
