package menger.geometry

import io.github.lene.optix.InstanceId

/** A fractional-level object drawn as its two whole levels, which [[MengerRenderer]] renders
  * separately and mixes: `lower` (the instance ids of level n) with weight 1 - fraction,
  * `upper` (level n+1) with weight fraction. Each render shows one level of every such object,
  * so no light path meets both levels of one object. Fading one level in the same render as
  * the other cannot work for glass: a refractive surface shows what lies behind it at any
  * strength (usability review 2026-10, session 4, F85; decided by user 2026-10-14: linear mix,
  * rendering twice). */
final case class LevelMix(lower: Vector[InstanceId], upper: Vector[InstanceId], fraction: Float):
  require(fraction > 0f && fraction < 1f, s"a level mix needs a fraction in (0, 1), got $fraction")

  def contains(instanceId: InstanceId): Boolean =
    lower.contains(instanceId) || upper.contains(instanceId)

object LevelMix:

  /** Up to this many mixed objects, every combination of their levels is rendered: 2^k
    * renders, exact. Beyond, the first-order sum (k + 1 renders), exact where no light path
    * meets two of the objects. */
  val MaxExactMixes = 4

  private val ByteMask = 0xff
  private val MaxChannel = 255L

  /** The renders one frame needs for objects with these fractions: per render, which objects
    * show their upper level, and the render's weight. */
  def worlds(fractions: Vector[Float]): Vector[(Vector[Boolean], Double)] =
    val k = fractions.size
    if k <= MaxExactMixes then
      (0 until (1 << k)).toVector.map { bits =>
        val upper = Vector.tabulate(k)(i => ((bits >> i) & 1) == 1)
        val weight = upper.zip(fractions).map((up, f) => if up then f.toDouble else 1.0 - f).product
        (upper, weight)
      }
    else
      // All lower, plus one render per object with only that object upper:
      // (1 - sum f) * lower + sum f_i * (only i upper) = lower + sum f_i * (difference i makes).
      (Vector.fill(k)(false), 1.0 - fractions.map(_.toDouble).sum) +:
        Vector.tabulate(k)(i => (Vector.tabulate(k)(_ == i), fractions(i).toDouble))

  /** The weighted sum of same-size RGBA images, rounded and clamped to 0..255 (the first-order
    * sum can weight a render negatively). optix-jni's `renderMix` mixes the same way on the
    * native side; this is the oracle its result is tested against. */
  def mix(images: Seq[(Array[Byte], Double)]): Array[Byte] =
    val length = images.head._1.length
    require(images.forall(_._1.length == length), "all images of a level mix must have one size")
    val sum = new Array[Double](length)
    images.foreach { (image, weight) =>
      (0 until length).foreach(i => sum(i) += (image(i) & ByteMask) * weight)
    }
    sum.map(v => math.max(0L, math.min(MaxChannel, math.round(v))).toByte)
