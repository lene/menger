package examples.dsl

import scala.language.implicitConversions

import menger.dsl._
import menger.spec.PlaneRotation
import menger.spec.ProjectionSpec

/**
 * Example: a tesseract lifted into 5D and turned there (menger#162). Its x y z w coordinates stay
 * and the fifth axis v starts at 0, so at t = 0 the picture is the tesseract seen through two
 * perspective stages (5D -> 4D -> 3D). Turning in the xv plane then tips x into v; turning in
 * wv after it does the same for w -- a rotation no 4D view can show, because the shape is not
 * rotated inside the 4D space the `Projection4DSpec` projects.
 *
 * `planes` are applied in the order written: xv first, then wv. Each stage of the projection
 * scales a flat object by (eye - screen) / eye, so with two stages `size` is chosen larger than
 * for a 4D tesseract of the same look, and `screen` is low.
 *
 * Usage:
 *   --scene examples.dsl.TesseractToPenteract --t 3
 */
object TesseractToPenteract:
  val durationSeconds = 8f

  private val turnSeconds = durationSeconds / 2f
  private val turnDegrees = 60f
  private val eye = 6f
  private val screen = 1.2f
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
        PlaneRotation("wv", progress(t, turnSeconds) * turnDegrees)
      ),
      eye = eye,
      screen = screen
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
