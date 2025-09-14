package menger.objects.higher_d

import menger.objects.{FixedVector, Vector}

class Edge(val v0: Vector[4], val v1: Vector[4]) extends FixedVector[2, Vector[4]](v0, v1):
  override def toString: String = s"${v0.toString} -> ${v1.toString}"
  def asSeq: Seq[Vector[4]] = values.toSeq
  def diff: Vector[4] = v1 - v0
