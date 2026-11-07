package examples.dsl

import scala.language.implicitConversions

import menger.dsl._
import menger.spec.PlaneRotation
import menger.spec.ProjectionSpec
import menger.spec.ProjectionStage

/**
 * Example: the hexeract, the cube of six dimensions, turning in a plane with its sixth axis u
 * (menger#163). It has 64 vertices, 192 edges (gold tubes) and 240 square faces (thin film), and
 * is seen through three perspective stages, 6D -> 5D -> 4D -> 3D.
 *
 * The cube has an extent in every axis, so a turn in the xu plane is a real turn: u moves into
 * x. After a quarter turn the cube maps onto itself, which makes the 8 s loop seamless.
 *
 * Usage:
 *   --scene examples.dsl.HexeractTurning --t 2
 */
object HexeractTurning:
  val durationSeconds = 8f

  private val quarterTurn = 90f
  private val stages = List.fill(3)(ProjectionStage(4f, 0.8f))
  private val size = 1.6f
  private val material = Some(Material.Film)
  private val edgeRadius = Some(0.012f)
  private val edgeMaterial = Some(Material.Gold)

  def scene(t: Float): Scene =
    val view = ProjectionSpec(
      dimension = 6,
      planes = List(PlaneRotation("xu", quarterTurn * t / durationSeconds)),
      stages = stages
    )
    Scene(
      camera = Camera(position = (2.5f, 2f, 4f), lookAt = (0f, 0f, 0f)),
      objects = List(
        Hypercube(6, material = material, size = size, edgeRadius = edgeRadius,
          edgeMaterial = edgeMaterial, projection = Some(view))
      ),
      lights = List(
        Directional(direction = (1f, -1f, -1f), intensity = 1.5f),
        Directional(direction = (-1f, -0.5f, 1f), intensity = 0.5f)
      ),
      planes = List(Plane(Y at -2, color = "#303030"))
    )
