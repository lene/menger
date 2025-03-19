package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4

class Edge(val v0: Vector4, val v1: Vector4) extends FixedVector[2, Vector4](v0, v1):
  override def toString: String = s"${v0.asString} -> ${v1.asString}"
  def asSeq: Seq[Vector4] = values
  def diff: Vector4 = v1 - v0
