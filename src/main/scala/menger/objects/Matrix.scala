package menger.objects

import com.badlogic.gdx.math.Matrix4
import menger.Const

case class Matrix[dim <: Int & Singleton](m: Array[Float])(implicit d: ValueOf[dim]):
  lazy val dimension: Int = d.value
  require(m.length == d.value * d.value, s"Expected ${d.value * d.value} elements, got ${m.length}")

  def === (that: Matrix[dim]): Boolean = epsilonEquals(that)

  def mul(that: Matrix[dim]): Matrix[dim] =
    val result = Array.fill(d.value * d.value)(0f)
    for (i <- 0 until d.value; j <- 0 until d.value) {
      for (k <- 0 until d.value) {
        result(index(i, j)) += this(i, k) * that(k, j)
      }
    }
    Matrix[dim](result)

  def apply(i: Int, j: Int): Float =
    m(index(i, j))

  def index(i: Int, j: Int): Int =
    require(i >= 0 && i < d.value && j >= 0 && j < d.value, s"Indices must be between 0 and ${d.value - 1}, got ($i, $j)")
    i * d.value + j

  def apply(v: Vector[dim]): Vector[dim] =
    val m = (0 until d.value).map { i =>
      (0 until d.value).foldLeft(0f) { (acc, j) => acc + this(i, j) * v(j) }
    }
    Vector.fromSeq[dim](m)

  override def toString: String =
    val bdArray = m.map("% 2.2f".format(_))
    val mArray = (0 until d.value).map { i =>
      (0 until d.value).map { j => bdArray(index(i, j)) }.mkString("|", " ", "|")
    }
    mArray.mkString("\n")

  def epsilonEquals(that: Matrix[dim]): Boolean =
    that.m.zip(m).forall((a, b) => math.abs(a - b) < Const.epsilon)

case object Matrix:
  def identity[dim <: Int & Singleton](implicit d: ValueOf[dim]): Matrix[dim] =
    val size = d.value
    val values = Array.fill(size * size)(0f)
    for (i <- 0 until size) values(i * size + i) = 1f
    Matrix(values)(using d)
