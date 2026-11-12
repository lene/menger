package examples.dsl

import scala.language.implicitConversions

import menger.dsl._
import menger.spec.PlaneRotation
import menger.spec.ProjectionSpec
import menger.spec.ProjectionStage

/**
 * Example: the penteract, the cube of five dimensions, unfolding along its newest axis v and
 * folding back (menger#164). Its `scale` along v grows from 0 to 1 and back to 0, eased by a
 * smoothstep; at 0 the penteract is flat in v, a tesseract in the 5D view. This shows the
 * per-axis scale, not the hand-over to a separate lower object (that is `CubeToTesseract`).
 *
 * It turns in the plane xv, which mixes the new axis into the old ones, and in the plane xz,
 * which is a whole turn of the 3D picture. Both make a full 360 degrees over the loop, so the
 * loop is seamless, and the plane involving v is at a multiple of 180 degrees when v is flat.
 *
 * Usage:
 *   --scene examples.dsl.PenteractFold --t 6
 */
object PenteractFold:
  val durationSeconds = 12f

  private val fullTurn = 360f
  // Two projection stages (5D -> 4D -> 3D), each with its eye above the cube's extent along the
  // axis it drops and the screen close to the origin, so the picture stays large.
  private val stageEye = 4f
  private val stageScreen = 0.8f
  private val stageCount = 2
  private val stages = List.fill(stageCount)(ProjectionStage(stageEye, stageScreen))
  private val size = 1.6f
  // Film has an IOR of 1.33, which sends the shader down the refractive path; IOR 1 is a thin
  // film without refraction.
  private val material = Some(Material.Film.copy(ior = 1.0f))
  private val edgeRadius = Some(0.012f)
  private val edgeMaterial = Some(Material.Gold)

  private def smoothstep(x: Float): Float =
    val clamped = math.min(1f, math.max(0f, x))
    clamped * clamped * (3f - 2f * clamped)

  /** The time within the loop, also for t outside 0 to `durationSeconds`. */
  private def loopTime(t: Float): Float =
    ((t % durationSeconds) + durationSeconds) % durationSeconds

  /** 0 at the start and the end of the loop, 1 in the middle. */
  def extrusion(t: Float): Float =
    smoothstep(1f - math.abs(2f * loopTime(t) / durationSeconds - 1f))

  def scene(t: Float): Scene =
    val turn = fullTurn * loopTime(t) / durationSeconds
    val view = ProjectionSpec(
      dimension = 5,
      planes = List(PlaneRotation("xv", turn), PlaneRotation("xz", turn)),
      stages = stages
    )
    Scene(
      camera = Camera(position = (2.5f, 2f, 4f), lookAt = (0f, 0f, 0f)),
      objects = List(
        Hypercube(5, material = material, size = size, edgeRadius = edgeRadius,
          edgeMaterial = edgeMaterial, projection = Some(view),
          scale = Some(Scale.identity[5].withLast(extrusion(t))))
      ),
      lights = List(
        Directional(direction = (1f, -1f, -1f), intensity = 1.5f),
        Directional(direction = (-1f, -0.5f, 1f), intensity = 0.5f)
      ),
      planes = List(Plane(Y at -2, color = "#303030"))
    )
