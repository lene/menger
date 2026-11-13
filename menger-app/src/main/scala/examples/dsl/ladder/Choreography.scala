package examples.dsl.ladder

import scala.language.implicitConversions

import menger.dsl._
import menger.objects.higher_d.NdSpongeVariant
import menger.spec.PlaneRotation
import menger.spec.ProjectionSpec
import menger.spec.ProjectionStage

/** The timeline of one cycle: `depths` maps a dimension to its breath depth (a missing one is 1). */
final case class Choreography(cycle: Float, top: Int, depths: Map[Int, Int]):
  import Choreography._

  require(cycle > 0f, s"the cycle must be positive, got $cycle")
  require(top >= minTop && top <= maxTop, s"topDimension must be 4 to 8, got $top")

  // The costed levels, capped above 3D by the surface sponge's triangle budget.
  def breathDepth(n: Int): Int =
    require(n >= 3 && n <= maxTop, s"breathDepth needs a dimension from 3 to $maxTop, got $n")
    val costed = depths.getOrElse(n, 1)
    if n == 3 then costed
    else math.min(costed, ResourceLimits.ndSpongeLevel(NdSpongeVariant.Surface, n).max)

  private def ascentWeights: List[(Kind, Int, Float)] =
    val perDimension = (3 to top).toList.flatMap: n =>
      List(
        Option.when(n >= minTop)((Kind.Fold, n, foldSeconds)),
        Option.when(n >= minTop)((Kind.Solidify, n, solidifySeconds)),
        Some((Kind.Breath, n, breathDepth(n) * breathSecondsPerLevel)),
        Option.when(n < top)((Kind.Unsolidify, n, unsolidifySeconds))
      ).flatten
    (Kind.Hold, 3, holdSeconds) :: perDimension

  val segments: List[Segment] =
    val descent = cycle * descentFraction
    val ascent = ascentWeights
    val stretch = (cycle - descent) / ascent.map(_._3).sum
    val timed = ascent.map((kind, n, weight) => (kind, n, weight * stretch)) ++
      (top to minTop by -1).map(n => (Kind.Collapse, n, descent / (top - 3)))
    val starts = timed.map(_._3).scanLeft(0f)(_ + _)
    timed.zip(starts).map { case ((kind, n, seconds), start) => Segment(kind, n, start, seconds) }

  def phaseOf(t: Float): Float = ((t % cycle) + cycle) % cycle

  /** The segment at `phase` in the cycle, and how far into it, from 0 to 1. */
  def segmentAt(phase: Float): (Segment, Float) =
    val segment = segments.findLast(_.start <= phase).getOrElse(segments.head)
    (segment, math.max(0f, math.min(1f, (phase - segment.start) / segment.seconds)))

  // The ascent planes between the new axis and x, y, z wake in the fold and settle at a half turn
  // before the next fold; the descent planes among the extra axes turn while their axis collapses.
  val turns: List[Turn] =
    def startOf(kind: Kind, n: Int): Option[Float] =
      segments.find(s => s.kind == kind && s.dimension == n).map(_.start)
    val descentStart = segments.find(_.kind == Kind.Collapse).fold(cycle)(_.start)
    val wake = for
      n <- (minTop to top).toList
      partner <- wakingPartners
    yield Turn(s"$partner${axisLetters(n - 1)}", startOf(Kind.Fold, n).getOrElse(0f),
      startOf(Kind.Fold, n + 1).getOrElse(descentStart))
    val settle = for
      n <- (top to minTop + 1 by -1).toList
      collapse <- segments.find(s => s.kind == Kind.Collapse && s.dimension == n).toList
      partner <- firstExtraAxis until n - 1
    yield Turn(s"${axisLetters(partner)}${axisLetters(n - 1)}", collapse.start, collapse.end)
    wake ++ settle

  /** A whole number of turns per cycle in the xz plane: the 3D turntable. */
  def turntableDegrees(phase: Float): Float =
    fullTurn * turntableTurns * phase / cycle

  def planeAngles(phase: Float): List[PlaneRotation] =
    turns.map(turn => PlaneRotation(turn.plane, turn.degrees(phase))) :+
      PlaneRotation("xz", turntableDegrees(phase))

object Choreography:
  val axisLetters = "xyzwvuts"
  val descentFraction = 1f / 5f

  enum Kind:
    case Hold, Fold, Solidify, Breath, Unsolidify, Collapse

  case class Segment(kind: Kind, dimension: Int, start: Float, seconds: Float):
    def end: Float = start + seconds

  case class Turn(plane: String, from: Float, to: Float):
    def degrees(phase: Float): Float = halfTurn * smoothstep((phase - from) / (to - from))

  private val holdSeconds = 2f
  private val foldSeconds = 6f
  private val solidifySeconds = 2f
  private val unsolidifySeconds = 2f
  private val breathSecondsPerLevel = 4f
  private val halfTurn = 180f
  private val fullTurn = 360f
  private val turntableTurns = 2
  private val wakingPartners = "xyz"
  private val firstExtraAxis = 3
  private[ladder] val minTop = 4
  private val maxTop = 8
  // The stage dropping axis k has its eye and screen at these multiples of the object's size
  // there, so every stage shrinks and the 3D -> 4D size rule s * eye / (eye - screen) holds for
  // each further dimension. A stage keeps its values once its axis exists.
  private val eyeRatio = 4f
  private val screenRatio = 0.8f
  private val growthPerDimension = eyeRatio / (eyeRatio - screenRatio)
  // A fold or collapse this flat is drawn as the lower object: a near-zero axis would draw
  // every face twice, and the two shapes differ by less than this fraction of the size.
  private[ladder] val standInExtrusion = 1e-3f
  private[ladder] val baseSize = 1f

  private[ladder] def smoothstep(x: Float): Float =
    val clamped = math.min(1f, math.max(0f, x))
    clamped * clamped * (3f - 2f * clamped)

  private[ladder] def bell(u: Float): Float = ((1.0 - math.cos(2.0 * math.Pi * u)) / 2.0).toFloat

  private[ladder] def sizeOf(n: Int): Float =
    baseSize * math.pow(growthPerDimension, n - 3).toFloat

  // The stages of the n-view, the first dropping the highest axis; the tail is the (n-1)-view's.
  private def stagesFor(n: Int): List[ProjectionStage] =
    (n to minTop by -1).toList.map: k =>
      ProjectionStage(eyeRatio * sizeOf(k), screenRatio * sizeOf(k))

  private def stageFactor(stage: ProjectionStage, coordinate: Float): Float =
    require(coordinate < stage.eye, s"coordinate $coordinate reaches the eye at ${stage.eye}")
    (stage.eye - stage.screen) / (stage.eye - coordinate)

  /** The projected distance of the (+h, ..., +h) corner from the centre, with the newest axis
    * scaled by `last`: continuous at every hand-over, so a dolly keeps the object's size in
    * frame. */
  def halfExtent(n: Int, last: Float): Float =
    val h = sizeOf(n) / 2f
    val projected = stagesFor(n).foldLeft(List.fill(n - 1)(h) :+ h * last): (p, stage) =>
      p.init.map(_ * stageFactor(stage, p.last))
    math.sqrt(projected.map(x => x * x).sum).toFloat

  /** An upper bound of the projected distance from the centre of any point of the n-cube, its
    * newest axis scaled by `last`, turned in any plane: the circumscribed radius, carried through
    * the stages. A stage drops the coordinate c of a point at distance r; the rest, at most
    * sqrt(r^2 - c^2) away, is scaled by (eye - screen) / (eye - c), which peaks at c = r^2 / eye.
    * The first stage caps c at `last * r`: a flat axis gives the (n-1)-cube's bound. */
  def boundingRadius(n: Int, last: Float = 1f): Float =
    val start = sizeOf(n) / 2f * math.sqrt(n - 1 + last * last).toFloat
    stagesFor(n).zipWithIndex.foldLeft(start): (r, stageAt) =>
      val (stage, index) = stageAt
      require(r < stage.eye, s"radius $r reaches the eye at ${stage.eye}")
      val c = math.min(r * r / stage.eye, if index == 0 then last * r else r)
      (stage.eye - stage.screen) * math.sqrt(r * r - c * c).toFloat / (stage.eye - c)

  private[ladder] def extrusion(segment: Segment, u: Float): Float = segment.kind match
    case Kind.Fold     => smoothstep(u)
    case Kind.Collapse => 1f - smoothstep(u)
    case _             => 1f

  // `Scale` needs its dimension as a type, which a runtime n cannot give; `n.type` stands in,
  // and `Scale` itself checks that it got n factors.
  private[ladder] def scaleOf(n: Int, last: Float): Scale[?] =
    Scale[n.type]((List.fill(n - 1)(1f) :+ last)*)(using ValueOf(n))

  private[ladder] def viewOf(n: Int, planes: List[PlaneRotation]): ProjectionSpec =
    ProjectionSpec(n, planes.filter(_.plane.highestAxis < n), stages = stagesFor(n))
