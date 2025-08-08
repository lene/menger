package menger.objects

import menger.Const

/// An n-dimensional vector
case class Vector[dim <: Int & Singleton, T <: AnyVal](v: Float*)(implicit d: ValueOf[dim]):

  lazy val dimension: Int = d.value
  require(v.size == d.value, s"Expected ${d.value} elements, got ${v.size}")

  def + (delta: Vector[dim, T]): Vector[dim, T] =
    Vector.fromSeq(v.zip(delta.v).map { case (a, b) => a + b })(using d)

  def - (that: Vector[dim, T]): Vector[dim, T] =
    Vector.fromSeq(v.zip(that.v).map { case (a, b) => a - b })(using d)

  def unary_- : Vector[dim, T] =
    Vector.fromSeq(v.map(-_))(using d)

  def === (that: Vector[dim, T]): Boolean =
    v.zip(that.v).forall { case (a, b) => math.abs(a - b) < Const.epsilon }

  def len(): Float = // todo remove brackets, kept for backwards compatibility
    math.sqrt(v.map(x => x * x).sum).toFloat

  def * (scalar: Float): Vector[dim, T] =
    Vector.fromSeq(v.map(x => x * scalar))(using d)

  def / (scalar: Float): Vector[dim, T] = *(1 / scalar)

  def dst(that: Vector[dim, T]): Float =
    (this - that).len()

  def dst2(that: Vector[dim, T]): Float =
    v.zip(that.v).map { case (a, b) => (a - b) * (a - b) }.sum

  def dot(that: Vector[dim, T]): Float =
    v.zip(that.v).foldLeft(0f) { case (acc, (a, b)) => acc + a * b }

  @deprecated("use indexWhere directly on v","07-08-2025")
  lazy val toArray: Array[Float] = v.toArray

  def x: Float = v(0)
  def y: Float = v(1)
  def z: Float = v(2)
  def w: Float = v(3)

  def asString: String = vec2string(this)

  def epsilonEquals(that: Vector[dim, T], epsilon: Float = Const.epsilon): Boolean =
    v.zip(that.v).forall { case (a, b) => math.abs(a - b) < epsilon }


case object Vector:
  def fromSeq[dim <: Int & Singleton, T <: AnyVal](seq: Seq[Float])(implicit d: ValueOf[dim]): Vector[dim, T] =
    Vector(seq*)(using d)

  def Zero[dim <: Int & Singleton, T <: AnyVal](implicit d: ValueOf[dim]): Vector[dim, T] =
    Vector(Array.fill(d.value)(0f)*)(using d)

  val X: Vector[4, Float] = Vector[4, Float](1f, 0f, 0f, 0f)
  val Y: Vector[4, Float] = Vector[4, Float](0f, 1f, 0f, 0f)
  val Z: Vector[4, Float] = Vector[4, Float](0f, 0f, 1f, 0f)
  val W: Vector[4, Float] = Vector[4, Float](0f, 0f, 0f, 1f)

type Vector4 = Vector[4, Float]

def vec2string[dim <: Int & Singleton, T <: AnyVal](vec: Vector[dim, T]): String = vec.toArray.map(float2string).mkString("<", ", ", ">")
def float2string(f: Float): String = if (f - f.toInt).abs > 1e-6 then f"$f% .2f" else s"${f.toInt}"
