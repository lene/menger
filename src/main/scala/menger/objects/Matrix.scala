package menger.objects

import menger.Const

case class Matrix[dim <: Int & Singleton](m: Array[Float])(implicit d: ValueOf[dim]):

  require(d.value > 0, "Matrix dimension must be positive")
  require(m.length == d.value * d.value, s"Expected ${d.value * d.value} elements, got ${m.length}")
  def dimension: Int = d.value
  lazy val indices: Seq[Int] = 0 until d.value

  def === (that: Matrix[dim]): Boolean = epsilonEquals(that)

  def *(that: Matrix[dim]): Matrix[dim] =
    val result = Array.fill(m.length)(0f)
    def element(i: Int, j: Int) = indices.map(k => this(i, k) * that(k, j)).sum
    for (i <- indices; j <- indices) result(index(i, j)) = element(i, j)
    Matrix[dim](result)

  def apply(i: Int, j: Int): Float = m(index(i, j))

  def index(i: Int, j: Int): Int =
    require(areValidIndices(i, j), s"Indices must be between 0 and ${d.value - 1}, got ($i, $j)")
    i * d.value + j

  private def areValidIndices(i: Int, j: Int) = i >= 0 && i < d.value && j >= 0 && j < d.value

  def apply(v: Vector[dim]): Vector[dim] =
    def element(i: Int) = indices.foldLeft(0f) { (acc, j) => acc + this (i, j) * v(j) }
    Vector[dim](indices.map(element)*)

  override def toString: String =
    val asStrings = m.map("% 2.2f".format(_))
    def rowString(i: Int) = indices.map { j => asStrings(index(i, j)) }.mkString("|", " ", "|")
    val rows = indices.map(rowString)
    rows.mkString("\n")

  def epsilonEquals(that: Matrix[dim]): Boolean =
    that.m.zip(m).forall((a, b) => math.abs(a - b) < Const.epsilon)


case object Matrix:
  def identity[dim <: Int & Singleton](implicit d: ValueOf[dim]): Matrix[dim] =
    val size = d.value
    val values = Array.fill(size * size)(0f)
    for (i <- 0 until size) values(i * size + i) = 1f
    Matrix[dim](values)


