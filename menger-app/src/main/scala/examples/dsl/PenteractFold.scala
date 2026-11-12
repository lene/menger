package examples.dsl

import scala.language.implicitConversions

import menger.dsl._
import menger.spec.PlaneRotation
import menger.spec.ProjectionSpec
import menger.spec.ProjectionStage

/**
 * Example: the penteract, the cube of five dimensions, unfolding out of the tesseract and folding
 * back into it (menger#164). Its `scale` along the newest axis v grows from 0 to 1 and back to
 * 0, eased by a smoothstep; at 0 the penteract is the tesseract, and nothing else of the scene
 * changes, so the dimension ladder's hand-over needs no cut.
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
  private val stages = List.fill(2)(ProjectionStage(4f, 0.8f))
  private val size = 1.6f
  private val material = Some(Material.Film)
  private val edgeRadius = Some(0.012f)
  private val edgeMaterial = Some(Material.Gold)

  private def smoothstep(x: Float): Float = x * x * (3f - 2f * x)

  /** 0 at the start and the end of the loop, 1 in the middle. */
  def extrusion(t: Float): Float =
    smoothstep(1f - math.abs(2f * t / durationSeconds - 1f))

  def scene(t: Float): Scene =
    val turn = fullTurn * t / durationSeconds
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
