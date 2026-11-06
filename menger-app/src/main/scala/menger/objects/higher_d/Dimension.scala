package menger.objects.higher_d

/** A dimension known only at runtime, with its type-level counterpart `N` and the `ValueOf[N]`
  * the generic types need: `val d = Dimension.of(n); given ValueOf[d.N] = d.valueOf`. */
sealed abstract class Dimension(val value: Int):
  type N <: Int & Singleton
  def valueOf: ValueOf[N]

object Dimension:
  val min: Int = 3
  val axisNames: IndexedSeq[Char] = "xyzwvuts"

  private final class Case[A <: Int & Singleton](using v: ValueOf[A]) extends Dimension(v.value):
    type N = A
    def valueOf: ValueOf[N] = v

  // one case per supported dimension, from `min` upwards
  private val supported: Vector[Dimension] =
    Vector(new Case[3], new Case[4], new Case[5], new Case[6], new Case[7], new Case[8])

  val max: Int = min + supported.size - 1

  def isSupported(n: Int): Boolean = n >= min && n <= max

  private[higher_d] def requireSupported(n: Int, context: String): Unit =
    require(isSupported(n), s"dimension must be between $min and $max, is $n ($context)")

  def of(n: Int): Dimension =
    requireSupported(n, "Dimension.of")
    supported(n - min)
