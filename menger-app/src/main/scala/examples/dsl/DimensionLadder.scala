package examples.dsl

import scala.language.implicitConversions

import menger.dsl._
import menger.objects.higher_d.NdSpongeVariant
import menger.spec.PlaneRotation
import menger.spec.ProjectionSpec
import menger.spec.ProjectionStage

/**
 * Example: the dimension ladder (menger#160), a looping animation. A metal cube breathes into a
 * Menger sponge, folds into 4D, breathes, folds into 5D and so on up to `topDimension`, then
 * descends to the cube along a different path (planes among the extra axes only), under a
 * sunrise with one matching sun. A segment timeline (`segments`) drives everything, and
 * `scene(t)` depends on t only through its phase in the cycle, so scene(0) == scene(T).
 * The breath is a Menger sponge in every dimension: the surface sponge (menger#166) above 3D.
 *
 * Usage: --scene examples.dsl.DimensionLadder --t 30
 */
object DimensionLadder:
  val cycleSeconds = 180f
  val durationSeconds = cycleSeconds
  // The top of the ladder is E4's (menger#168); 4 to 8 work.
  val topDimension = 6
  // The tint, a hue from the highest dimension reached, is E8's (menger#172).
  val tint = Color(0.78f, 0.82f, 0.95f)
  val axisLetters = "xyzwvuts"
  val descentFraction = 1f / 5f
  // The costed levels, capped above 3D by the surface sponge's triangle budget.
  def breathDepth(n: Int): Int = n match
    case 3 => 5
    case 4 => math.min(3, surfaceLimit(4))
    case 5 => math.min(2, surfaceLimit(5))
    case _ => math.min(1, surfaceLimit(n))
  private def surfaceLimit(n: Int): Int =
    ResourceLimits.ndSpongeLevel(NdSpongeVariant.Surface, n).max

  enum Kind:
    case Hold, Fold, Solidify, Breath, Unsolidify, Collapse

  case class Segment(kind: Kind, dimension: Int, start: Float, seconds: Float):
    def end: Float = start + seconds

  case class Turn(plane: String, from: Float, to: Float):
    def degrees(phase: Float): Float = halfTurn * smoothstep((phase - from) / (to - from))

  // Timeline weights in seconds, scaled so that the ascent fills the cycle but the descent.
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
  private val minTop = 4
  private val maxTop = 8
  // The stage dropping axis k has its eye and screen at these multiples of the object's size
  // there, so every stage shrinks and the 3D -> 4D size rule s * eye / (eye - screen) holds for
  // each further dimension. A stage keeps its values once its axis exists.
  private val eyeRatio = 4f
  private val screenRatio = 0.8f
  private val growthPerDimension = eyeRatio / (eyeRatio - screenRatio)
  // A fold or collapse this flat is drawn as the lower object: a near-zero axis would draw
  // every face twice, and the two shapes differ by less than this fraction of the size.
  private val standInExtrusion = 1e-3f
  private val baseSize = 1f
  private val edgeRadius = 0.012f
  private val filmRamp = 0.25f
  private val metal = Material.metal(tint)
  private val film = Material.Film.copy(ior = 1.0f)
  // The render constants, copied from scripts/frame-budget/Moment3D.scala: G4 measures them.
  private val envMap = "assets/hdri/kiara_2_sunrise_4k.hdr"
  private val keyLight = Sun(
    direction = Vec3(-0.7423f, -0.0722f, -0.6662f),
    diameterDegrees = 0.53f,
    color = Color(1.00f, 0.86f, 0.40f)
  )
  private val plinthTopY = -0.45f
  private val plinthThickness = 0.2f
  private val plinthRadius = 1.8f
  private val plinthColor = Color(0.85f, 0.82f, 0.76f)
  // 0.01 above the plinth's top cap: a bottom face on the cap itself speckles the refraction.
  private val plinthClearance = 0.01f
  private val causticsPhotons = 200000
  private val causticsIterations = 10
  // Low, on the far side from the sun (along its rays), 15 degrees to one side.
  private val cameraDistance = 4.5f
  private val cameraYawDegrees = 15f
  private val cameraBelowCentre = -0.15f
  private val cameraDirection =
    val yaw = math.toRadians(cameraYawDegrees)
    val (x, z) = (keyLight.direction.x, keyLight.direction.z)
    Vec3((x * math.cos(yaw) + z * math.sin(yaw)).toFloat, cameraBelowCentre,
      (-x * math.sin(yaw) + z * math.cos(yaw)).toFloat).normalize

  private def smoothstep(x: Float): Float =
    val clamped = math.min(1f, math.max(0f, x))
    clamped * clamped * (3f - 2f * clamped)

  private def bell(u: Float): Float = ((1.0 - math.cos(2.0 * math.Pi * u)) / 2.0).toFloat

  private def mix(a: Material, b: Material, t: Float): Material =
    def lerp(x: Float, y: Float): Float = x * (1f - t) + y * t
    Material(
      Color(lerp(a.color.r, b.color.r), lerp(a.color.g, b.color.g), lerp(a.color.b, b.color.b),
        lerp(a.color.a, b.color.a)),
      lerp(a.ior, b.ior), lerp(a.roughness, b.roughness), lerp(a.metallic, b.metallic),
      lerp(a.specular, b.specular), lerp(a.emission, b.emission),
      lerp(a.filmThickness, b.filmThickness), lerp(a.dispersion, b.dispersion))

  def phaseOf(t: Float, cycle: Float = cycleSeconds): Float =
    require(cycle > 0f, s"the cycle must be positive, got $cycle")
    ((t % cycle) + cycle) % cycle

  private def ascentWeights(top: Int): List[(Kind, Int, Float)] =
    val perDimension = (3 to top).toList.flatMap: n =>
      List(
        Option.when(n >= minTop)((Kind.Fold, n, foldSeconds)),
        Option.when(n >= minTop)((Kind.Solidify, n, solidifySeconds)),
        Some((Kind.Breath, n, breathDepth(n) * breathSecondsPerLevel)),
        Option.when(n < top)((Kind.Unsolidify, n, unsolidifySeconds))
      ).flatten
    (Kind.Hold, 3, holdSeconds) :: perDimension

  def segments(cycle: Float = cycleSeconds, top: Int = topDimension): List[Segment] =
    require(cycle > 0f, s"the cycle must be positive, got $cycle")
    require(top >= minTop && top <= maxTop, s"topDimension must be 4 to 8, got $top")
    val descent = cycle * descentFraction
    val ascent = ascentWeights(top)
    val stretch = (cycle - descent) / ascent.map(_._3).sum
    val timed = ascent.map((kind, n, weight) => (kind, n, weight * stretch)) ++
      (top to minTop by -1).map(n => (Kind.Collapse, n, descent / (top - 3)))
    val starts = timed.map(_._3).scanLeft(0f)(_ + _)
    timed.zip(starts).map { case ((kind, n, seconds), start) => Segment(kind, n, start, seconds) }

  private val defaultTimeline = segments()

  private def timeline(cycle: Float, top: Int): List[Segment] =
    if cycle == cycleSeconds && top == topDimension then defaultTimeline else segments(cycle, top)

  /** The segment at `phase` in the cycle, and how far into it, from 0 to 1. */
  def segmentAt(phase: Float, cycle: Float = cycleSeconds, top: Int = topDimension)
      : (Segment, Float) =
    val all = timeline(cycle, top)
    val segment = all.findLast(_.start <= phase).getOrElse(all.head)
    (segment, math.max(0f, math.min(1f, (phase - segment.start) / segment.seconds)))

  // The ascent planes between the new axis and x, y, z wake in the fold and settle at a half
  // turn before the next fold. The descent planes among the extra axes turn while their axis
  // collapses, so every plane is at a multiple of 180 degrees when its axis is gone.
  def turns(cycle: Float = cycleSeconds, top: Int = topDimension): List[Turn] =
    val all = timeline(cycle, top)
    def startOf(kind: Kind, n: Int): Option[Float] =
      all.find(s => s.kind == kind && s.dimension == n).map(_.start)
    val descentStart = all.find(_.kind == Kind.Collapse).fold(cycle)(_.start)
    val wake = for
      n <- (minTop to top).toList
      partner <- wakingPartners
    yield Turn(s"$partner${axisLetters(n - 1)}", startOf(Kind.Fold, n).getOrElse(0f),
      startOf(Kind.Fold, n + 1).getOrElse(descentStart))
    val settle = for
      n <- (top to minTop + 1 by -1).toList
      collapse <- all.find(s => s.kind == Kind.Collapse && s.dimension == n).toList
      partner <- firstExtraAxis until n - 1
    yield Turn(s"${axisLetters(partner)}${axisLetters(n - 1)}", collapse.start, collapse.end)
    wake ++ settle

  /** A whole number of turns per cycle in the xz plane: the 3D turntable. */
  def turntableDegrees(phase: Float, cycle: Float = cycleSeconds): Float =
    fullTurn * turntableTurns * phase / cycle

  def planeAngles(phase: Float, cycle: Float = cycleSeconds, top: Int = topDimension)
      : List[PlaneRotation] =
    turns(cycle, top).map(turn => PlaneRotation(turn.plane, turn.degrees(phase))) :+
      PlaneRotation("xz", turntableDegrees(phase, cycle))

  private def sizeOf(n: Int): Float = baseSize * math.pow(growthPerDimension, n - 3).toFloat

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

  /** An upper bound of the projected distance from the centre of any point of the n-cube turned
    * in any plane: the circumscribed radius, carried through the stages. A stage drops the
    * coordinate c of a point at distance r; the rest, at most sqrt(r^2 - c^2) away, is scaled by
    * (eye - screen) / (eye - c), which peaks at c = r^2 / eye. */
  def boundingRadius(n: Int): Float =
    stagesFor(n).foldLeft(sizeOf(n) / 2f * math.sqrt(n).toFloat): (r, stage) =>
      require(r < stage.eye, s"radius $r reaches the eye at ${stage.eye}")
      (stage.eye - stage.screen) * r / math.sqrt(stage.eye * stage.eye - r * r).toFloat

  private def centre(top: Int): Vec3 =
    val widest = (3 to top).map(boundingRadius).max + edgeRadius
    Vec3(0f, plinthTopY + plinthClearance + widest, 0f)

  private def extrusion(segment: Segment, u: Float): Float = segment.kind match
    case Kind.Fold     => smoothstep(u)
    case Kind.Collapse => 1f - smoothstep(u)
    case _             => 1f

  private case class Skin(material: Material, edgeRadius: Option[Float])

  // 0 is solid metal, 1 is film with edge tubes.
  private def skin(filminess: Float): Skin =
    Skin(mix(metal, film, filminess), Option.when(filminess > 0f)(edgeRadius * filminess))

  // The descent starts from the metal breath, so its first stretch grows the film, and the
  // last collapse lands as the metal cube.
  private def collapseFilm(n: Int, u: Float, top: Int): Float =
    val grown = if n == top then smoothstep(u / filmRamp) else 1f
    val solidified = if n == minTop then 1f - smoothstep(u) else 1f
    grown * solidified

  private def turntable(planes: List[PlaneRotation]): Vec3 =
    val degrees = planes.find(_.plane.letters == "xz").fold(0f)(_.angle.value)
    Vec3(0f, math.toRadians(degrees).toFloat, 0f)

  // `Scale` needs its dimension as a type, which a runtime n cannot give; `n.type` stands in,
  // and `Scale` itself checks that it got n factors.
  private def scaleOf(n: Int, last: Float): Scale[?] =
    Scale[n.type]((List.fill(n - 1)(1f) :+ last)*)(using ValueOf(n))

  private def viewOf(n: Int, planes: List[PlaneRotation]): ProjectionSpec =
    ProjectionSpec(n, planes.filter(_.plane.highestAxis < n), stages = stagesFor(n))

  private def hypercubeOf(n: Int, skin: Skin, last: Float, planes: List[PlaneRotation], top: Int)
      : SceneObject =
    if n == 3 then
      Cube(pos = centre(top), material = Some(skin.material), size = baseSize,
        edgeRadius = skin.edgeRadius, edgeMaterial = skin.edgeRadius.map(_ => metal),
        rotation = turntable(planes))
    else
      val view = viewOf(n, planes)
      Hypercube(n, pos = centre(top), material = Some(skin.material), size = sizeOf(n),
        edgeRadius = skin.edgeRadius, edgeMaterial = skin.edgeRadius.map(_ => metal),
        projection = Some(view), scale = Some(scaleOf(n, last)))

  private def breathObject(n: Int, level: Float, material: Material,
      planes: List[PlaneRotation], top: Int): SceneObject =
    if n == 3 then
      Sponge(VolumeFilling, pos = centre(top), level = level, material = Some(material),
        size = baseSize, rotation = turntable(planes))
    else
      HypercubeSponge(n, TesseractSpongeType.SurfaceSubdividing, level, pos = centre(top),
        material = Some(material), size = sizeOf(n), projection = Some(viewOf(n, planes)))

  /** The object in `segment`, `u` of the way through. While the newest axis of a fold or
    * collapse is flatter than `standInExtrusion`, the lower object stands in: it is the same
    * shape, and the higher object would draw its faces twice. */
  def objectAt(segment: Segment, u: Float, cycle: Float = cycleSeconds, top: Int = topDimension)
      : SceneObject =
    val planes = planeAngles(segment.start + u * segment.seconds, cycle, top)
    val n = segment.dimension
    val last = extrusion(segment, u)
    val (shown, shownLast) = if last <= standInExtrusion then (n - 1, 1f) else (n, last)
    segment.kind match
      case Kind.Hold => hypercubeOf(3, skin(0f), 1f, planes, top)
      case Kind.Breath =>
        breathObject(n, breathDepth(n) * bell(u), mix(metal, Material.Glass, bell(u)), planes, top)
      case Kind.Solidify => hypercubeOf(n, skin(1f - smoothstep(u)), 1f, planes, top)
      case Kind.Unsolidify => hypercubeOf(n, skin(smoothstep(u)), 1f, planes, top)
      case Kind.Fold => hypercubeOf(shown, skin(1f), shownLast, planes, top)
      case Kind.Collapse =>
        hypercubeOf(shown, skin(collapseFilm(n, u, top)), shownLast, planes, top)

  def sceneAt(t: Float, cycle: Float, top: Int): Scene =
    val (segment, u) = segmentAt(phaseOf(t, cycle), cycle, top)
    val dolly = halfExtent(segment.dimension, extrusion(segment, u)) / halfExtent(3, 1f)
    val target = centre(top)
    Scene(
      camera = Camera(position = target + cameraDirection * (cameraDistance * dolly),
        lookAt = target),
      objects = List(
        Cylinder(
          p0 = Some(Vec3(0f, plinthTopY - plinthThickness, 0f)),
          p1 = Some(Vec3(0f, plinthTopY, 0f)),
          radius = Some(plinthRadius),
          material = Some(Material.matte(plinthColor))
        ),
        objectAt(segment, u, cycle, top)
      ),
      lights = List(keyLight),
      envMap = Some(envMap),
      ibl = Some(IBL()),
      caustics = Some(Caustics(photonsPerIteration = causticsPhotons,
        iterations = causticsIterations)),
      // User, 2026-11-12: "Yes, acc16 now"; accumulation does not smooth edges until
      // optix-jni#116, and adaptive antialiasing costs ~140x on glass (optix-jni#115).
      render = Some(RenderSettings(antialiasing = false, shadows = true, accumulation = 16))
    )

  def scene(t: Float): Scene = sceneAt(t, cycleSeconds, topDimension)
