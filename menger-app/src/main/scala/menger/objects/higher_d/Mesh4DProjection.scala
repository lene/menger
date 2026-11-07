package menger.objects.higher_d

import menger.common.Vector

/** Renders any 4D mesh by projecting it to 3D space.
  *
  * @param mesh4D The 4D mesh to project
  * @param center Center position of the projected mesh in 3D space
  * @param eyeW Distance from eye to projection hyperplane in 4D (must be > screenW)
  * @param screenW Distance from origin to projection hyperplane in 4D (must be > 0)
  * @param rotXW Rotation angle in XW plane (degrees)
  * @param rotYW Rotation angle in YW plane (degrees)
  * @param rotZW Rotation angle in ZW plane (degrees)
  */
case class Mesh4DProjection(
    mesh4D: Mesh4D,
    center: Vector[3] = Vector[3](0f, 0f, 0f),
    eyeW: Float = 3.0f,
    screenW: Float = 1.5f,
    rotXW: Float = 15f,
    rotYW: Float = 10f,
    rotZW: Float = 0f
) extends ProjectedMesh4D:

  require(eyeW > screenW, s"eyeW ($eyeW) must be greater than screenW ($screenW)")
  require(eyeW > 0 && screenW > 0, "eyeW and screenW must be positive")

  private val rotation: Rotation[4] =
    if rotXW == 0f && rotYW == 0f && rotZW == 0f then Rotation.identity
    else Rotation(rotXW, rotYW, rotZW, Vector[4](0f, 0f, 0f, 0f))

  private val projection = Projection(eyeW, screenW)

  override protected def projectVertex(vertex: Vector[4]): Vector[3] =
    projection(rotation(vertex))

/** Backward-compatible factory for creating tesseract meshes */
object TesseractMesh:
  def apply(
      center: Vector[3] = Vector[3](0f, 0f, 0f),
      size: Float = 1.0f,
      eyeW: Float = 3.0f,
      screenW: Float = 1.5f,
      rotXW: Float = 15f,
      rotYW: Float = 10f,
      rotZW: Float = 0f
  ): Mesh4DProjection =
    Mesh4DProjection(
      mesh4D = Tesseract(size = size),
      center = center,
      eyeW = eyeW,
      screenW = screenW,
      rotXW = rotXW,
      rotYW = rotYW,
      rotZW = rotZW
    )
