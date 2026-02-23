package examples.dsl

import scala.language.implicitConversions

import menger.Projection4DSpec
import menger.dsl._

/**
 * Example: 4D Tesseract rendering
 *
 * This demonstrates rendering a 4D hypercube (tesseract) projected into 3D space.
 *
 * The scene features:
 * - A tesseract with custom 4D projection parameters
 * - Glass material to show inner structure
 * - Optional edge rendering for wireframe view
 *
 * The 4D object is projected from 4D space to 3D by placing a virtual camera
 * in 4D (eyeW) looking at a 3D hyperplane (screenW = 0).
 *
 * Usage: --scene examples.dsl.TesseractDemo
 */
object TesseractDemo:
  val scene = Scene(
    camera = Camera(
      position = (0f, 2f, 5f),
      lookAt = (0f, 0f, 0f)
    ),
    objects = List(
      Tesseract(Material.Glass).copy(
        projection = Some(Projection4DSpec(
          eyeW = 3.0f,
          screenW = 1.5f,
          rotXW = 15f,
          rotYW = 10f,
          rotZW = 0f
        ))
      )
    ),
    lights = List(
      Directional(
        direction = (1f, -1f, -1f),
        intensity = 1.5f
      )
    )
  )

  SceneRegistry.register("tesseract-demo", scene)
