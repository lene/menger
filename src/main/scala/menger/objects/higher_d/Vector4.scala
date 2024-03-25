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
  def toArray: Array[Float] = Array(v.x, v.y, v.z, v.w)
