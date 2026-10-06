package examples.dsl

import scala.language.implicitConversions

import menger.Projection4DSpec
import menger.dsl._

/**
 * Example: the regular 4D polytopes besides the tesseract, side by side.
 *
 * Pentachoron (5-cell), Hexadecachoron (16-cell), Icositetrachoron (24-cell),
 * Hecatonicosachoron (120-cell) and Hexacosichoron (600-cell), each with glass faces and thin
 * edges. All edge-rendered 4D objects in one scene must share one 4D projection.
 *
 * Usage: --scene examples.dsl.PolytopeGallery
 */
object PolytopeGallery:
  private val projection = Some(Projection4DSpec(eyeW = 3.0f, screenW = 1.5f, rotXW = 20f, rotYW = 15f))
  private val edges = Some(0.01f)

  val scene = Scene(
    camera = Camera(
      position = (0f, 4f, 20f),
      lookAt = (0f, 0f, 0f)
    ),
    objects = List(
      Pentachoron(pos = (-6f, 0f, 0f), material = Some(Material.Glass), projection = projection,
        edgeRadius = edges),
      Hexadecachoron(pos = (-3f, 0f, 0f), material = Some(Material.Glass), projection = projection,
        edgeRadius = edges),
      Icositetrachoron(pos = (0f, 0f, 0f), material = Some(Material.Glass),
        projection = projection, edgeRadius = edges),
      Hecatonicosachoron(pos = (3f, 0f, 0f), material = Some(Material.Glass),
        projection = projection, edgeRadius = edges),
      Hexacosichoron(pos = (6f, 0f, 0f), material = Some(Material.Glass), projection = projection,
        edgeRadius = edges)
    ),
    lights = List(
      Directional(
        direction = (1f, -1f, -1f),
        intensity = 1.5f
      )
    ),
    planes = List(Plane(Y at -1.5, color = "#FFFFFF"))
  )

  SceneRegistry.register("polytope-gallery", scene)
