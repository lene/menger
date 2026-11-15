package examples.dsl.stills

import scala.language.implicitConversions

import menger.dsl._
import menger.spec.PlaneRotation
import menger.spec.ProjectionSpec
import menger.spec.ProjectionStage

/**
 * Still pictures of the penteract, the hexeract and their sponges, each object on its own, as
 * references outside the animations. One object, turned in the planes that mix the extra axes
 * into x, y and z, over a checkered floor.
 *
 * The cubes are a soap film with gold edges, so every cell of the nested cubes shows. The
 * sponges are a blue-tinted glass: an opaque sponge of five or six dimensions hides its inner
 * cells and looks like a 4D sponge in perspective; the glass shows them, and the tint gives the
 * nested layers depth against the floor.
 *
 * In a sub-package, so they stay out of the scene corpus, whose scenes are self-contained.
 *
 * Usage: --scene examples.dsl.stills.PenteractStill (and the other objects below)
 */
object HypercubeStills:
  /** w, v and u turned into x, y and z: one extra axis for each old one. */
  private val turns =
    List(PlaneRotation("xw", 25f), PlaneRotation("yv", 20f), PlaneRotation("zu", 15f))
  // One projection stage per dimension above 3, each eye above the object's extent along the
  // axis it drops, the screen close to the origin, so the picture stays large.
  private val stage = ProjectionStage(4f, 0.8f)
  private val size = 1.6f
  private val edgeRadius = 0.012f
  private val spongeGlass = Material.glass(Color(0.55f, 0.8f, 1.0f))
  private val floorLight = Color(0.75f, 0.75f, 0.75f)
  private val floorDark = Color(0.15f, 0.15f, 0.15f)

  private def view(dimension: Int): ProjectionSpec =
    ProjectionSpec(dimension = dimension, planes = turns.take(dimension - 3),
      stages = List.fill(dimension - 3)(stage))

  /** The n-cube as a film, its edges in gold. */
  def cube(dimension: Int): Scene = scene(
    Hypercube(dimension, material = Some(Material.Film), size = size,
      edgeRadius = Some(edgeRadius), edgeMaterial = Some(Material.Gold),
      projection = Some(view(dimension))))

  def sponge(dimension: Int, spongeType: TesseractSpongeType, level: Float): Scene = scene(
    HypercubeSponge(dimension, spongeType, level, material = Some(spongeGlass), size = size,
      projection = Some(view(dimension))))

  private def scene(obj: SceneObject): Scene = Scene(
    camera = Camera(position = (2.5f, 2f, 4f), lookAt = (0f, 0f, 0f)),
    objects = List(obj),
    lights = List(
      Directional(direction = (1f, -1f, -1f), intensity = 1.5f),
      Directional(direction = (-1f, -0.5f, 1f), intensity = 0.5f)
    ),
    // A neutral sky: the glass mirrors it on the faces turned up.
    background = Some(Color(0.55f, 0.6f, 0.68f)),
    planes = List(Plane.checkered(Y at -2, (floorLight, floorDark)))
  )

object PenteractStill:
  val scene: Scene = HypercubeStills.cube(5)

object HexeractStill:
  val scene: Scene = HypercubeStills.cube(6)

object PenteractSpongeSurface1:
  val scene: Scene = HypercubeStills.sponge(5, TesseractSpongeType.SurfaceSubdividing, 1f)

object PenteractSpongeSurface2:
  val scene: Scene = HypercubeStills.sponge(5, TesseractSpongeType.SurfaceSubdividing, 2f)

object PenteractSpongeVolume1:
  val scene: Scene = HypercubeStills.sponge(5, TesseractSpongeType.VolumeRemoving, 1f)

object HexeractSpongeSurface1:
  val scene: Scene = HypercubeStills.sponge(6, TesseractSpongeType.SurfaceSubdividing, 1f)

object HexeractSpongeSurface2:
  val scene: Scene = HypercubeStills.sponge(6, TesseractSpongeType.SurfaceSubdividing, 2f)

object HexeractSpongeVolume1:
  val scene: Scene = HypercubeStills.sponge(6, TesseractSpongeType.VolumeRemoving, 1f)
