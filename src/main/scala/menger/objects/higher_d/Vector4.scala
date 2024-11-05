package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4

import scala.annotation.targetName

extension (v: Vector4)
  @targetName("times")
  def *(a: Float): Vector4 = Vector4(v.x * a, v.y * a, v.z * a, v.w * a)
  @targetName("dividedBy")
  def /(a: Float): Vector4 = v * (1 / a)
  @targetName("plus")
  def +(v2: Vector4): Vector4 = Vector4(v.x + v2.x, v.y + v2.y, v.z + v2.z, v.w + v2.w)
  def -(v2: Vector4): Vector4 = Vector4(v.x - v2.x, v.y - v2.y, v.z - v2.z, v.w - v2.w)
  def toArray: Array[Float] = Array(v.x, v.y, v.z, v.w)
//  def toString: String = f"(${v.x}%.2f, ${v.y}%.2f, ${v.z}%.2f, ${v.w}%.2f)"
  def toVec2: String = f"(${v.z}%.2f, ${v.w}%.2f)"

/// An n-dimensional vector TODO: continue later
case class Vector[dim <: Int & Singleton](v: Float*)(implicit d: ValueOf[dim]):
  lazy val dimension: Int = d.value
  assert(v.size == dimension, s"Expected $dimension elements, got ${v.size}")
  