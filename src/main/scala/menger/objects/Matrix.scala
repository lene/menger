package menger.objects

import com.badlogic.gdx.math.Matrix4

case class Matrix[dim <: Int & Singleton, T <: AnyVal](m: Array[Float])(implicit d: ValueOf[dim]):
  lazy val dimension: Int = d.value
  require(m.size == d.value * d.value, s"Expected ${d.value * d.value} elements, got ${m.size}")

  lazy val asArray: Array[Float] = m

  def mul(that: Matrix[dim, T]): Matrix[dim, T] =
    val result = Array.fill(d.value * d.value)(0f)
    for (i <- 0 until d.value; j <- 0 until d.value) {
      for (k <- 0 until d.value) {
        result(index(i, j)) += this(i, k) * that(k, j)
      }
    }
    Matrix(result)(using d)

  def apply(i: Int, j: Int): Float =
    m(index(i, j))

  def index(i: Int, j: Int): Int =
    require(i >= 0 && i < d.value && j >= 0 && j < d.value, s"Indices must be between 0 and ${d.value - 1}, got ($i, $j)")
    i * d.value + j

  def apply(v: Vector[dim, T]): Vector[dim, T] =
    val m0 = this(0, 0) * v.x + this(0, 1) * v.y + this(0, 2) * v.z + this(0, 3) * v.w
    val m1 = this(1, 0) * v.x + this(1, 1) * v.y + this(1, 2) * v.z + this(1, 3) * v.w
    val m2 = this(2, 0) * v.x + this(2, 1) * v.y + this(2, 2) * v.z + this(2, 3) * v.w
    val m3 = this(3, 0) * v.x + this(3, 1) * v.y + this(3, 2) * v.z + this(3, 3) * v.w
    val v_ = Vector[dim, T](m0, m1, m2, m3)
    v_

  def str: String =
    val bdArray = m.map("% 2.2f".format(_))
    val mArray = Array(
      Array(bdArray(index(0, 0)), bdArray(index(0, 1)), bdArray(index(0, 2)), bdArray(index(0, 3))).mkString("|", " ", "|"),
      Array(bdArray(index(1, 0)), bdArray(index(1, 1)), bdArray(index(1, 2)), bdArray(index(1, 3))).mkString("|", " ", "|"),
      Array(bdArray(index(2, 0)), bdArray(index(2, 1)), bdArray(index(2, 2)), bdArray(index(2, 3))).mkString("|", " ", "|"),
      Array(bdArray(index(3, 0)), bdArray(index(3, 1)), bdArray(index(3, 2)), bdArray(index(3, 3))).mkString("|", " ", "|"),
    )
    mArray.mkString("\n")

case object Matrix:
  def identity[dim <: Int & Singleton, T <: AnyVal](implicit d: ValueOf[dim]): Matrix[dim, T] =
    val size = d.value
    val values = Array.fill(size * size)(0f)
    for (i <- 0 until size) values(i * size + i) = 1f
    Matrix(values)(using d)
