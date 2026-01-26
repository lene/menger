package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3

/** Factory for creating surface-based 4D Menger sponge meshes (tesseract-sponge-2).
  *
  * TesseractSponge2 generates a 4D sponge by recursive surface subdivision,
  * similar to the 3D Menger sponge surface approach. At each level, each face
  * is divided into smaller faces with the center removed, creating a hollow
  * fractal structure.
  *
  * Face count growth: 24 * 16^level
  * - Level 0: 24 faces (basic tesseract)
  * - Level 1: 384 faces
  * - Level 2: 6,144 faces
  * - Level 3: 98,304 faces
  * - Level 4: 1,572,864 faces
  * - Level 5: 25,165,824 faces
  */
object TesseractSponge2Mesh:

  /** Create a surface-based 4D sponge mesh projected to 3D space.
    *
    * @param center Center position of the projected mesh in 3D space
    * @param size Size of the sponge
    * @param level Recursion depth (0 = tesseract, 1+ = sponge)
    * @param eyeW Distance from eye to projection hyperplane in 4D (must be > screenW)
    * @param screenW Distance from origin to projection hyperplane in 4D (must be > 0)
    * @param rotXW Rotation angle in XW plane (degrees)
    * @param rotYW Rotation angle in YW plane (degrees)
    * @param rotZW Rotation angle in ZW plane (degrees)
    * @return Mesh4DProjection ready for rendering via OptiX
    */
  def apply(
      center: Vector3 = Vector3(0f, 0f, 0f),
      size: Float = 1.0f,
      level: Float,
      eyeW: Float = 3.0f,
      screenW: Float = 1.5f,
      rotXW: Float = 15f,
      rotYW: Float = 10f,
      rotZW: Float = 0f
  ): Mesh4DProjection =
    Mesh4DProjection(
      mesh4D = TesseractSponge2(level, size),
      center = center,
      eyeW = eyeW,
      screenW = screenW,
      rotXW = rotXW,
      rotYW = rotYW,
      rotZW = rotZW
    )

  /** Estimate number of 4D faces for a given level.
    *
    * Formula: 24 * 16^level
    * where 24 is the base tesseract face count and 16 is the branching factor
    * (each face subdivides into 8 smaller faces + 8 perpendicular faces).
    *
    * @param level Recursion depth
    * @return Estimated number of 4D quad faces
    */
  def estimatedFaces(level: Int): Long =
    if level == 0 then 24L
    else 24L * math.pow(16, level).toLong

  /** Estimate number of triangles after projection to 3D.
    *
    * Each 4D quad face projects to a 3D quad, which is tessellated into 2 triangles.
    *
    * @param level Recursion depth
    * @return Estimated number of triangles
    */
  def estimatedTriangles(level: Int): Long =
    estimatedFaces(level) * 2
