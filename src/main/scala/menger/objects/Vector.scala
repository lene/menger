package menger.objects

import menger.Const


case class Vector[dim <: Int & Singleton](v: Float*)(implicit d: ValueOf[dim]):

  lazy val dimension: Int = d.value
  require(v.size == d.value, s"Expected ${d.value} elements, got ${v.size}")

  def apply(i: Int): Float =
    require(i >= 0 && i < d.value, s"Index must be between 0 and ${d.value - 1}, got $i")
    v(i)
    
  def + (delta: Vector[dim]): Vector[dim] =
    Vector.fromSeq(v.zip(delta.v).map { case (a, b) => a + b })(using d)

  def - (that: Vector[dim]): Vector[dim] =
    Vector.fromSeq(v.zip(that.v).map { case (a, b) => a - b })(using d)

  def unary_- : Vector[dim] =
    Vector.fromSeq(v.map(-_))(using d)

  def === (that: Vector[dim]): Boolean = epsilonEquals(that)

  def len: Float = 
    math.sqrt(v.map(x => x * x).sum).toFloat

  def * (scalar: Float): Vector[dim] =
    Vector.fromSeq(v.map(x => x * scalar))(using d)

  def / (scalar: Float): Vector[dim] = *(1 / scalar)

  def dst(that: Vector[dim]): Float =
    math.sqrt(dst2(that)).toFloat

  def dst2(that: Vector[dim]): Float =
    v.zip(that.v).map { case (a, b) => (a - b) * (a - b) }.sum

  def dot(that: Vector[dim]): Float =
    v.zip(that.v).foldLeft(0f) { case (acc, (a, b)) => acc + a * b }
  
  override def toString: String = v.map(float2string).mkString("<", ", ", ">")

  def epsilonEquals(that: Vector[dim], epsilon: Float = Const.epsilon): Boolean =
    v.zip(that.v).forall { case (a, b) => math.abs(a - b) < epsilon }

  def count(p: Float => Boolean): Int = v.count(p)
  def filter(p: Float => Boolean): Seq[Float] = v.filter(p)
  def forall(p: Float => Boolean): Boolean = v.forall(p)
  def indexWhere(p: Float => Boolean, from: Int = 0): Int = v.indexWhere(p, from)
  def map[B](f: Float => B): Seq[B] = v.map(f)
  def toIndexedSeq: IndexedSeq[Float] = v.toIndexedSeq
  
  
case object Vector:
  def fromSeq[dim <: Int & Singleton](seq: Seq[Float])(implicit d: ValueOf[dim]): Vector[dim] =
    Vector(seq*)(using d)

  def Zero[dim <: Int & Singleton](implicit d: ValueOf[dim]): Vector[dim] =
    Vector(Array.fill(d.value)(0f)*)(using d)

  val X: Vector[4] = Vector[4](1f, 0f, 0f, 0f)
  val Y: Vector[4] = Vector[4](0f, 1f, 0f, 0f)
  val Z: Vector[4] = Vector[4](0f, 0f, 1f, 0f)
  val W: Vector[4] = Vector[4](0f, 0f, 0f, 1f)


def float2string(f: Float): String = 
  if (f - f.toInt).abs > Const.epsilon then f"$f% .2f" else s"${f.toInt}"
