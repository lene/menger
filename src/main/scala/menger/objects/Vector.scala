package menger.objects

import menger.Const


case class Vector[dim <: Int & Singleton](v: Float*)(implicit d: ValueOf[dim]):

  require(d.value > 0, "Vector dimension must be positive")
  require(v.size == d.value, s"Expected ${d.value} elements, got ${v.size}")
  def dimension: Int = d.value

  def apply(i: Int): Float =
    require(i >= 0 && i < d.value, s"Index must be between 0 and ${d.value - 1}, got $i")
    v(i)
    
  def + (delta: Vector[dim]): Vector[dim] =
    Vector[dim](v.zip(delta.v).map { case (a, b) => a + b }*)

  def - (that: Vector[dim]): Vector[dim] =
    Vector[dim](v.zip(that.v).map { case (a, b) => a - b }*)

  def unary_- : Vector[dim] = Vector[dim](v.map(-_)*)

  def === (that: Vector[dim]): Boolean = epsilonEquals(that)

  def epsilonEquals(that: Vector[dim], epsilon: Float = Const.epsilon): Boolean =
    v.zip(that.v).forall { case (a, b) => math.abs(a - b) < epsilon }

  def len2: Float = v.map(x => x * x).sum

  def len: Float = math.sqrt(len2).toFloat

  def * (that: Vector[dim]): Float =
    v.zip(that.v).foldLeft(0f) { case (acc, (a, b)) => acc + a * b }

  def * (scalar: Float): Vector[dim] = Vector[dim](v.map(x => x * scalar)*)

  def / (scalar: Float): Vector[dim] = *(1 / scalar)

  def dst2(that: Vector[dim]): Float = (that - this).len2

  def dst(that: Vector[dim]): Float = (that - this).len

  override def toString: String = v.map(float2string).mkString("<", ", ", ">")

  def count(p: Float => Boolean): Int = v.count(p)
  def filter(p: Float => Boolean): Seq[Float] = v.filter(p)
  def forall(p: Float => Boolean): Boolean = v.forall(p)
  def indexWhere(p: Float => Boolean, from: Int = 0): Int = v.indexWhere(p, from)
  def map[B](f: Float => B): Seq[B] = v.map(f)
  def toIndexedSeq: IndexedSeq[Float] = v.toIndexedSeq

  
case object Vector:
  def fromSeq[dim <: Int & Singleton](seq: Seq[Float])(implicit d: ValueOf[dim]): Vector[dim] =
    Vector[dim](seq*)

  def Zero[dim <: Int & Singleton](implicit d: ValueOf[dim]): Vector[dim] =
    Vector[dim](Array.fill(d.value)(0f)*)

  def unit[dim <: Int & Singleton](direction: Int)(implicit d: ValueOf[dim]): Vector[dim] =
    /** create a unit vector in the ith direction */
    require(0 <= direction && direction < d.value)
    val vec = Array.fill(d.value)(0f)
    vec(direction) = 1
    Vector[dim](vec*)

  val X: Vector[4] = unit[4](0)
  val Y: Vector[4] = unit[4](1)
  val Z: Vector[4] = unit[4](2)
  val W: Vector[4] = unit[4](3)


def float2string(f: Float): String = 
  if (f - f.toInt).abs > Const.epsilon then f"$f% .2f" else s"${f.toInt}"
