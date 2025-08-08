package menger.objects.higher_d

import menger.objects.Vector

class Edge(val v0: Vector[4, Float], val v1: Vector[4, Float]) extends FixedVector[2, Vector[4, Float]](v0, v1):
  override def toString: String = s"${v0.asString} -> ${v1.asString}"
  def asSeq: Seq[Vector[4, Float]] = values
  def diff: Vector[4, Float] = v1 - v0
