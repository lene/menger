package examples.dsl.ladder

import scala.language.implicitConversions

import examples.dsl.ladder.Choreography.baseSize
import examples.dsl.ladder.Choreography.bell
import examples.dsl.ladder.Choreography.extrusion
import examples.dsl.ladder.Choreography.minTop
import examples.dsl.ladder.Choreography.scaleOf
import examples.dsl.ladder.Choreography.sizeOf
import examples.dsl.ladder.Choreography.smoothstep
import examples.dsl.ladder.Choreography.standInExtrusion
import examples.dsl.ladder.Choreography.viewOf
import menger.dsl._
import menger.spec.PlaneRotation

/** Everything an experiment may vary without touching the choreography. */
final case class Look(
  metal: Material,
  film: Material,
  glass: Material,
  edgeRadius: Float,
  filmRamp: Float,
  keyLight: Sun,
  fillLights: List[Light],
  envMap: String,
  ibl: IBL,
  caustics: Caustics,
  render: RenderSettings,
  depths: Map[Int, Int]
)

object Look:
  val default: Look = Look(
    metal = Material.metal(DimensionLadder.tint),
    film = Material.Film.copy(ior = 1.0f),
    glass = Material.Glass,
    edgeRadius = 0.012f,
    filmRamp = 0.25f,
    // The render constants, copied from scripts/frame-budget/Moment3D.scala (G4 measures them).
    keyLight = Sun(direction = Vec3(-0.7423f, -0.0722f, -0.6662f), diameterDegrees = 0.53f,
      color = Color(1.00f, 0.86f, 0.40f)),
    fillLights = Nil,
    envMap = "assets/hdri/kiara_2_sunrise_4k.hdr",
    ibl = IBL(),
    caustics = Caustics(photonsPerIteration = 200000, iterations = 10),
    // Decided 2026-11-12: accumulation 16, no adaptive AA (~140x on glass, optix-jni#115);
    // revisit when optix-jni#116 (jitter) lands, as accumulation does not smooth edges.
    render = RenderSettings(antialiasing = false, shadows = true, accumulation = 16),
    // The costed breath levels, capped above 3D by the surface sponge's triangle budget.
    depths = Map(3 -> 5, 4 -> 3, 5 -> 2)
  )

/**
 * Example: the dimension ladder (menger#160), a looping animation. A metal cube breathes into a
 * Menger sponge, folds into 4D, breathes (a surface sponge above 3D, menger#166), folds into 5D
 * and so on up to `topDimension`, then descends to the cube along a different path (planes among
 * the extra axes only), under a sunrise with one matching sun. A segment timeline (`segments`)
 * drives everything, and `scene(t)` depends on t only through its phase: scene(0) == scene(T).
 * `sceneAt` takes a `Look` for the parts an experiment may vary (menger#184).
 *
 * The ladder lives in a sub-package, so it is loaded by name only (it stays out of the agent's
 * corpus): --scene examples.dsl.ladder.DimensionLadder --t 30
 */
object DimensionLadder:
  val cycleSeconds = 180f
  val durationSeconds = cycleSeconds
  val topDimension = 6 // E4 (menger#168) decides the top; 4 to 8 work
  val tint = Color(0.78f, 0.82f, 0.95f) // E8 (menger#172) derives it from the top dimension

  export Choreography.{Kind, Segment, Turn, axisLetters, descentFraction}

  private def choreography(cycle: Float, top: Int, look: Look): Choreography =
    Choreography(cycle, top, look.depths)

  private def stage(look: Look): Stage = Stage(look.keyLight.direction, look.edgeRadius)

  def segments(cycle: Float = cycleSeconds, top: Int = topDimension): List[Segment] =
    choreography(cycle, top, Look.default).segments

  def segmentAt(phase: Float, cycle: Float = cycleSeconds, top: Int = topDimension)
      : (Segment, Float) =
    choreography(cycle, top, Look.default).segmentAt(phase)

  def turns(cycle: Float = cycleSeconds, top: Int = topDimension): List[Turn] =
    choreography(cycle, top, Look.default).turns

  def planeAngles(phase: Float, cycle: Float = cycleSeconds, top: Int = topDimension)
      : List[PlaneRotation] =
    choreography(cycle, top, Look.default).planeAngles(phase)

  def phaseOf(t: Float, cycle: Float = cycleSeconds): Float =
    choreography(cycle, topDimension, Look.default).phaseOf(t)

  def breathDepth(n: Int): Int =
    choreography(cycleSeconds, topDimension, Look.default).breathDepth(n)

  def halfExtent(n: Int, last: Float): Float = Choreography.halfExtent(n, last)

  def boundingRadius(n: Int, last: Float = 1f): Float = Choreography.boundingRadius(n, last)

  private def mix(a: Material, b: Material, t: Float): Material =
    def lerp(x: Float, y: Float): Float = x * (1f - t) + y * t
    Material(
      Color(lerp(a.color.r, b.color.r), lerp(a.color.g, b.color.g), lerp(a.color.b, b.color.b),
        lerp(a.color.a, b.color.a)),
      lerp(a.ior, b.ior), lerp(a.roughness, b.roughness), lerp(a.metallic, b.metallic),
      lerp(a.specular, b.specular), lerp(a.emission, b.emission),
      lerp(a.filmThickness, b.filmThickness), lerp(a.dispersion, b.dispersion))

  private case class Skin(material: Material, edgeRadius: Option[Float],
      edgeMaterial: Option[Material])

  // 0 is solid metal, 1 is film with edge tubes.
  private def skin(filminess: Float, look: Look): Skin =
    Skin(mix(look.metal, look.film, filminess),
      Option.when(filminess > 0f)(look.edgeRadius * filminess),
      Option.when(filminess > 0f)(look.metal))

  // The descent starts from the metal breath, so its first stretch grows the film, and the
  // last collapse lands as the metal cube.
  private def collapseFilm(n: Int, u: Float, top: Int, look: Look): Float =
    val grown = if n == top then smoothstep(u / look.filmRamp) else 1f
    val solidified = if n == minTop then 1f - smoothstep(u) else 1f
    grown * solidified

  private def turntable(planes: List[PlaneRotation]): Vec3 =
    val degrees = planes.find(_.plane.letters == "xz").fold(0f)(_.angle.value)
    Vec3(0f, math.toRadians(degrees).toFloat, 0f)

  private def hypercubeOf(n: Int, skin: Skin, last: Float, planes: List[PlaneRotation], pos: Vec3)
      : SceneObject =
    if n == 3 then
      Cube(pos = pos, material = Some(skin.material), size = baseSize,
        edgeRadius = skin.edgeRadius, edgeMaterial = skin.edgeMaterial,
        rotation = turntable(planes))
    else
      val view = viewOf(n, planes)
      Hypercube(n, pos = pos, material = Some(skin.material), size = sizeOf(n),
        edgeRadius = skin.edgeRadius, edgeMaterial = skin.edgeMaterial,
        projection = Some(view), scale = Some(scaleOf(n, last)))

  private def breathObject(n: Int, level: Float, material: Material,
      planes: List[PlaneRotation], pos: Vec3): SceneObject =
    if n == 3 then
      Sponge(VolumeFilling, pos = pos, level = level, material = Some(material),
        size = baseSize, rotation = turntable(planes))
    else
      HypercubeSponge(n, TesseractSpongeType.SurfaceSubdividing, level, pos = pos,
        material = Some(material), size = sizeOf(n), projection = Some(viewOf(n, planes)))

  /** The object in `segment`, `u` of the way through. While the newest axis of a fold or
    * collapse is flatter than `standInExtrusion`, the lower object stands in: it is the same
    * shape, and the higher object would draw its faces twice. */
  def objectAt(segment: Segment, u: Float, cycle: Float = cycleSeconds, top: Int = topDimension,
      look: Look = Look.default): SceneObject =
    objectIn(choreography(cycle, top, look), segment, u, look)

  private def objectIn(timeline: Choreography, segment: Segment, u: Float, look: Look)
      : SceneObject =
    val planes = timeline.planeAngles(segment.start + u * segment.seconds)
    val n = segment.dimension
    val last = extrusion(segment, u)
    val pos = stage(look).centreAt(segment, u)
    val (shown, shownLast) = if last <= standInExtrusion then (n - 1, 1f) else (n, last)
    segment.kind match
      case Kind.Hold => hypercubeOf(3, skin(0f, look), 1f, planes, pos)
      case Kind.Breath =>
        breathObject(n, timeline.breathDepth(n) * bell(u),
          mix(look.metal, look.glass, bell(u)), planes, pos)
      case Kind.Solidify => hypercubeOf(n, skin(1f - smoothstep(u), look), 1f, planes, pos)
      case Kind.Unsolidify => hypercubeOf(n, skin(smoothstep(u), look), 1f, planes, pos)
      case Kind.Fold => hypercubeOf(shown, skin(1f, look), shownLast, planes, pos)
      case Kind.Collapse =>
        hypercubeOf(shown, skin(collapseFilm(n, u, timeline.top, look), look), shownLast,
          planes, pos)

  def sceneAt(t: Float, cycle: Float = cycleSeconds, top: Int = topDimension,
      look: Look = Look.default): Scene =
    val timeline = choreography(cycle, top, look)
    val (segment, u) = timeline.segmentAt(timeline.phaseOf(t))
    val stageOfLook = stage(look)
    Scene(
      camera = stageOfLook.cameraAt(segment, u),
      objects = List(stageOfLook.plinth, objectIn(timeline, segment, u, look)),
      lights = look.keyLight :: look.fillLights,
      envMap = Some(look.envMap),
      ibl = Some(look.ibl),
      caustics = Some(look.caustics),
      render = Some(look.render)
    )

  def scene(t: Float): Scene = sceneAt(t)
