package examples.dsl

import scala.language.implicitConversions

import menger.Projection4DSpec
import menger.dsl._

/**
 * Example: a cube extruded along w into a tesseract, then turned in 4D -- with no cut at the
 * hand-over (usability review 2026-10, session 5, F124, menger#132).
 *
 * What makes the hand-over seamless:
 *   - the 4D rotations stay 0 while `wScale` grows, so at wScale 0 the tesseract is the cube;
 *   - the tesseract's size is the cube's times eyeW / (eyeW - screenW): a flat 4D object is
 *     shown at (eyeW - screenW) / eyeW of its size;
 *   - both have the same material and the same edge tubes. A material, not just a colour: a
 *     4D object with edge tubes and no material is drawn as edges only;
 *   - the cube stands in for the tesseract at wScale exactly 0: there the tesseract's faces and
 *     edges lie on top of each other in pairs, and a transparent material speckles between
 *     the coincident copies. At any wScale above 0 it is the tesseract.
 *
 * Usage:
 *   --scene examples.dsl.CubeToTesseract --t 3
 */
object CubeToTesseract:
  val durationSeconds = 8f

  private val extrusionStart = 1f
  private val extrusionSeconds = 4f
  private val eyeW = 3f
  private val screenW = 1.5f
  private val cubeSize = 1f
  private val tesseractSize = cubeSize * eyeW / (eyeW - screenW)
  // Transparent, so the inner cube and the cells growing out of it show through.
  private val material = Some(Material.Film)
  private val edgeRadius = Some(0.015f)
  private val edgeMaterial = Some(Material.Gold)

  def scene(t: Float): Scene =
    val extruded = math.min(1f, math.max(0f, (t - extrusionStart) / extrusionSeconds))
    val turnStart = extrusionStart + extrusionSeconds
    val turn = math.max(0f, (t - turnStart) / (durationSeconds - turnStart)) * 90f
    val shape =
      if extruded == 0f then
        Cube(material = material, size = cubeSize, edgeRadius = edgeRadius,
          edgeMaterial = edgeMaterial)
      else
        Tesseract(material = material, size = tesseractSize, edgeRadius = edgeRadius,
          edgeMaterial = edgeMaterial,
          projection = Some(Projection4DSpec(eyeW = eyeW, screenW = screenW,
            rotXW = turn, wScale = extruded)))
    Scene(
      camera = Camera(position = (2.5f, 2f, 4f), lookAt = (0f, 0f, 0f)),
      objects = List(shape),
      lights = List(
        Directional(direction = (1f, -1f, -1f), intensity = 1.5f),
        Directional(direction = (-1f, -0.5f, 1f), intensity = 0.5f)
      ),
      planes = List(Plane(Y at -2, color = "#303030"))
    )
