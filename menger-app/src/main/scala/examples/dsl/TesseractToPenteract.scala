package examples.dsl

import scala.language.implicitConversions

import menger.dsl._
import menger.spec.PlaneRotation
import menger.spec.ProjectionSpec
import menger.spec.ProjectionStage

/**
 * Example: a tesseract lifted into 5D and turned there (menger#162). Its x y z w coordinates stay
 * and the fifth axis v starts at 0, so at t = 0 the picture is the tesseract seen through two
 * perspective stages (5D -> 4D -> 3D). It is turned in the xv plane, then in yv, then in zv, one
 * after the other and each by 45 degrees.
 *
 * Why this view: a tesseract has no extent in v, so turning it into v only changes its picture
 * through the perspective of the 5D -> 4D stage. That reads as a turn with a strong 5D
 * perspective (the 5D eye is close, 3) or with turns in several planes, which is why there are
 * three planes. The 5D stage's screen is 0.2 x its eye, so a flat object keeps its scale
 * ((eye - screen) / eye = 0.8 per stage) and `size` is chosen for the two stages together.
 *
 * `planes` are applied in the order written, and `progress` moves each plane in its own third
 * of the animation. `stages` lists the 5D -> 4D stage first.
 *
 * Usage:
 *   --scene examples.dsl.TesseractToPenteract --t 3
 */
object TesseractToPenteract:
  val durationSeconds = 8f

  private val turnSeconds = durationSeconds / 3f
  private val turnDegrees = 45f
  private val stages = List(ProjectionStage(3f, 0.6f), ProjectionStage(6f, 1.2f))
  private val size = 1.8f
  private val material = Some(Material.Film)
  private val edgeRadius = Some(0.015f)
  private val edgeMaterial = Some(Material.Gold)

  private def progress(t: Float, start: Float): Float =
    math.min(1f, math.max(0f, (t - start) / turnSeconds))

  def scene(t: Float): Scene =
    val view = ProjectionSpec(
      dimension = 5,
      planes = List(
        PlaneRotation("xv", progress(t, 0f) * turnDegrees),
        PlaneRotation("yv", progress(t, turnSeconds) * turnDegrees),
        PlaneRotation("zv", progress(t, 2f * turnSeconds) * turnDegrees)
      ),
      stages = stages
    )
    Scene(
      camera = Camera(position = (2.5f, 2f, 4f), lookAt = (0f, 0f, 0f)),
      objects = List(
        Tesseract(material = material, size = size, edgeRadius = edgeRadius,
          edgeMaterial = edgeMaterial, projection = Some(view))
      ),
      lights = List(
        Directional(direction = (1f, -1f, -1f), intensity = 1.5f),
        Directional(direction = (-1f, -0.5f, 1f), intensity = 0.5f)
      ),
      planes = List(Plane(Y at -2, color = "#303030"))
    )
