package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3

/** Factory for creating volume-based 4D Menger sponge meshes (tesseract-sponge).
  *
  * TesseractSponge generates a 4D sponge by recursive volume removal,
  * similar to the 3D Menger sponge approach. At each level, the tesseract
  * is divided into 3^4 = 81 sub-tesseracts, and interior cubes are removed.
  *
  * Face count growth: 24 * 48^level
  * - Level 0: 24 faces (basic tesseract)
  * - Level 1: 1,152 faces
  * - Level 2: 55,296 faces
  * - Level 3: 2,654,208 faces
  * - Level 4: 127,401,984 faces
  */
object TesseractSpongeMesh:

  /** Create a volume-based 4D sponge mesh projected to 3D space.
    *
    * @param center Center position of the projected mesh in 3D space
    * @param size Size of the sponge (unused for TesseractSponge, kept for API consistency)
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
      mesh4D = TesseractSponge(level),
      center = center,
      eyeW = eyeW,
      screenW = screenW,
      rotXW = rotXW,
      rotYW = rotYW,
      rotZW = rotZW
    )

  /** Estimate number of 4D faces for a given level.
    *
    * Formula: 24 * 48^level
    * where 24 is the base tesseract face count and 48 is the branching factor.
    *
    * @param level Recursion depth
    * @return Estimated number of 4D quad faces
    */
  def estimatedFaces(level: Int): Long =
    if level == 0 then 24L
    else 24L * math.pow(48, level).toLong

  /** Estimate number of triangles after projection to 3D.
    *
    * Each 4D quad face projects to a 3D quad, which is tessellated into 2 triangles.
    *
    * @param level Recursion depth
    * @return Estimated number of triangles
    */
  def estimatedTriangles(level: Int): Long =
    estimatedFaces(level) * 2
