package menger.objects

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import menger.common.TriangleMeshData
import menger.common.TriangleMeshSource

case class Octahedron(
  center: Vector3 = Vector3.Zero,
  scale: Float = 1f,
  material: Material = Builder.WHITE_MATERIAL,
  primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(center, scale) with TriangleMeshSource:

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  def getModel: List[ModelInstance] =
    throw UnsupportedOperationException("Octahedron is OptiX-only; getModel not supported")

  def toTriangleMesh: TriangleMeshData =
    val cx = center.x
    val cy = center.y
    val cz = center.z

    // Unit octahedron vertices inscribed in the unit sphere
    val v0x = 1f; val v0y = 0f; val v0z = 0f
    val v1x = -1f; val v1y = 0f; val v1z = 0f
    val v2x = 0f; val v2y = 1f; val v2z = 0f
    val v3x = 0f; val v3y = -1f; val v3z = 0f
    val v4x = 0f; val v4y = 0f; val v4z = 1f
    val v5x = 0f; val v5y = 0f; val v5z = -1f

    // For a regular octahedron inscribed in the unit sphere, vertex position == outward unit normal
    val rawVertices: Array[Float] = Array(
      v0x * scale + cx, v0y * scale + cy, v0z * scale + cz, v0x, v0y, v0z,
      v1x * scale + cx, v1y * scale + cy, v1z * scale + cz, v1x, v1y, v1z,
      v2x * scale + cx, v2y * scale + cy, v2z * scale + cz, v2x, v2y, v2z,
      v3x * scale + cx, v3y * scale + cy, v3z * scale + cz, v3x, v3y, v3z,
      v4x * scale + cx, v4y * scale + cy, v4z * scale + cz, v4x, v4y, v4z,
      v5x * scale + cx, v5y * scale + cy, v5z * scale + cz, v5x, v5y, v5z
    )

    // 8 triangular faces wound CCW from outside
    // Top cap (v2 = +y up): faces containing v2
    // Bottom cap (v3 = -y down): faces containing v3
    val indices: Array[Int] = Array(
      2, 4, 0,
      2, 1, 4,
      2, 5, 1,
      2, 0, 5,
      3, 0, 4,
      3, 4, 1,
      3, 1, 5,
      3, 5, 0
    )

    TriangleMeshData(rawVertices, indices, 6)
