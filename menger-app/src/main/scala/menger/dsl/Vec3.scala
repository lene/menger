package menger.dsl

import com.badlogic.gdx.math.{Vector3 => GdxVector3}
import menger.common.Vector

/** 3D vector for DSL with convenient tuple conversions */
case class Vec3(x: Float, y: Float, z: Float):
  def toGdxVector3: GdxVector3 = GdxVector3(x, y, z)
  def toCommonVector: Vector[3] = Vector[3](x, y, z)

object Vec3:
  val Zero = Vec3(0f, 0f, 0f)
  val UnitX = Vec3(1f, 0f, 0f)
  val UnitY = Vec3(0f, 1f, 0f)
  val UnitZ = Vec3(0f, 0f, 1f)

  // Implicit conversions for convenient tuple syntax
  given Conversion[(Float, Float, Float), Vec3] = t => Vec3(t._1, t._2, t._3)
  given Conversion[(Int, Int, Int), Vec3] = t => Vec3(t._1.toFloat, t._2.toFloat, t._3.toFloat)
  given Conversion[(Double, Double, Double), Vec3] = t => Vec3(t._1.toFloat, t._2.toFloat, t._3.toFloat)
