package examples.dsl

import scala.language.implicitConversions

import menger.dsl._
import menger.spec.PlaneRotation
import menger.spec.ProjectionSpec
import menger.spec.ProjectionStage

/**
 * Example: a surface sponge of five dimensions, built from the penteract, that breathes between
 * level 0 and level 2 (menger#166). At level 0 it is the penteract itself, in chrome; the level
 * rises smoothly to 2, the material turns from chrome through a grey plastic to glass, and it
 * falls back, so the loop is seamless. Level 2 has 28,160 faces; each whole level is built
 * once and kept, so a frame only projects it again.
 *
 * It turns 360 degrees in the plane xv, which mixes the fifth axis into the old ones, and 360
 * degrees in the plane xz, a whole turn of the 3D picture.
 *
 * Usage:
 *   --scene examples.dsl.PenteractSpongeBreath --t 8
 */
object PenteractSpongeBreath:
  val durationSeconds = 16f

  private val dimension = 5
  private val maxLevel = 2f
  private val fullTurn = 360f
  // Two projection stages (5D -> 4D -> 3D), each with its eye above the sponge's extent along the
  // axis it drops and the screen close to the origin, so the picture stays large.
  private val stageEye = 4f
  private val stageScreen = 0.8f
  private val stageCount = 2
  private val stages = List.fill(stageCount)(ProjectionStage(stageEye, stageScreen))
  private val size = 1.6f
  // The material changes in two halves of the level range: the metal first fades to a plain
  // grey surface at IOR 1, then the glass appears (IOR up, absorption down).
  private val halfWay = 0.5f
  private val glassIor = 1.5f
  private val glassAbsorption = 0.02f

  private def smoothstep(x: Float): Float =
    val clamped = math.min(1f, math.max(0f, x))
    clamped * clamped * (3f - 2f * clamped)

  private def lerp(from: Float, to: Float, x: Float): Float = from + (to - from) * x

  /** The time within the loop, also for t outside 0 to `durationSeconds`. */
  private def loopTime(t: Float): Float =
    ((t % durationSeconds) + durationSeconds) % durationSeconds

  /** 0 at the start and the end of the loop, `maxLevel` in the middle. */
  def level(t: Float): Float =
    maxLevel * smoothstep(1f - math.abs(2f * loopTime(t) / durationSeconds - 1f))

  /** Chrome at level 0, glass at `maxLevel`. */
  def material(level: Float): Material =
    val progress = level / maxLevel
    if progress <= halfWay then
      Material.Chrome.copy(metallic = 1f - progress / halfWay)
    else
      val glass = (progress - halfWay) / halfWay
      Material.Chrome.copy(
        color = Color(0.9f, 0.9f, 0.9f, lerp(1f, glassAbsorption, glass)),
        ior = lerp(1f, glassIor, glass),
        metallic = 0f
      )

  def scene(t: Float): Scene =
    val turn = fullTurn * loopTime(t) / durationSeconds
    val view = ProjectionSpec(
      dimension = dimension,
      planes = List(PlaneRotation("xv", turn), PlaneRotation("xz", turn)),
      stages = stages
    )
    val breath = level(t)
    Scene(
      camera = Camera(position = (2.5f, 2f, 4f), lookAt = (0f, 0f, 0f)),
      objects = List(
        HypercubeSponge(dimension, TesseractSpongeType.SurfaceSubdividing, breath,
          material = Some(material(breath)), size = size, projection = Some(view))
      ),
      lights = List(
        Directional(direction = (1f, -1f, -1f), intensity = 1.5f),
        Directional(direction = (-1f, -0.5f, 1f), intensity = 0.5f)
      ),
      planes = List(Plane(Y at -2, color = "#303030"))
    )
