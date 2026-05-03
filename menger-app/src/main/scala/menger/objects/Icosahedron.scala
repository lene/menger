package menger.objects

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import menger.common.TriangleMeshData
import menger.common.TriangleMeshSource

case class Icosahedron(
  center: Vector3 = Vector3.Zero,
  scale: Float = 1f,
  material: Material = Builder.WHITE_MATERIAL,
  primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(center, scale) with TriangleMeshSource:

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  def getModel: List[ModelInstance] =
    throw UnsupportedOperationException("Icosahedron is OptiX-only; getModel not supported")

  def toTriangleMesh: TriangleMeshData =
    val cx = center.x
    val cy = center.y
    val cz = center.z

    // Unit icosahedron vertices inscribed in the unit sphere
    // Based on three mutually perpendicular golden rectangles
    // φ = (1 + √5) / 2, normalized by √(1 + φ²)
    val v0x = 0f;          val v0y = 0.5257311f;  val v0z = 0.8506508f
    val v1x = 0f;          val v1y = -0.5257311f; val v1z = 0.8506508f
    val v2x = 0f;          val v2y = 0.5257311f;  val v2z = -0.8506508f
    val v3x = 0f;          val v3y = -0.5257311f; val v3z = -0.8506508f
    val v4x = 0.5257311f;  val v4y = 0.8506508f;  val v4z = 0f
    val v5x = -0.5257311f; val v5y = 0.8506508f;  val v5z = 0f
    val v6x = 0.5257311f;  val v6y = -0.8506508f; val v6z = 0f
    val v7x = -0.5257311f; val v7y = -0.8506508f; val v7z = 0f
    val v8x = 0.8506508f;  val v8y = 0f;          val v8z = 0.5257311f
    val v9x = -0.8506508f; val v9y = 0f;          val v9z = 0.5257311f
    val v10x = 0.8506508f; val v10y = 0f;         val v10z = -0.5257311f
    val v11x = -0.8506508f; val v11y = 0f;        val v11z = -0.5257311f

    // For a regular icosahedron inscribed in the unit sphere, vertex position == outward unit normal
    val rawVertices: Array[Float] = Array(
      v0x * scale + cx,  v0y * scale + cy,  v0z * scale + cz,  v0x,  v0y,  v0z,
      v1x * scale + cx,  v1y * scale + cy,  v1z * scale + cz,  v1x,  v1y,  v1z,
      v2x * scale + cx,  v2y * scale + cy,  v2z * scale + cz,  v2x,  v2y,  v2z,
      v3x * scale + cx,  v3y * scale + cy,  v3z * scale + cz,  v3x,  v3y,  v3z,
      v4x * scale + cx,  v4y * scale + cy,  v4z * scale + cz,  v4x,  v4y,  v4z,
      v5x * scale + cx,  v5y * scale + cy,  v5z * scale + cz,  v5x,  v5y,  v5z,
      v6x * scale + cx,  v6y * scale + cy,  v6z * scale + cz,  v6x,  v6y,  v6z,
      v7x * scale + cx,  v7y * scale + cy,  v7z * scale + cz,  v7x,  v7y,  v7z,
      v8x * scale + cx,  v8y * scale + cy,  v8z * scale + cz,  v8x,  v8y,  v8z,
      v9x * scale + cx,  v9y * scale + cy,  v9z * scale + cz,  v9x,  v9y,  v9z,
      v10x * scale + cx, v10y * scale + cy, v10z * scale + cz, v10x, v10y, v10z,
      v11x * scale + cx, v11y * scale + cy, v11z * scale + cz, v11x, v11y, v11z
    )

    // 20 triangular faces wound CCW from outside
    val indices: Array[Int] = Array(
      // Top ring (5 faces)
      0, 8, 4,
      0, 1, 8,
      0, 4, 5,
      0, 5, 9,
      0, 9, 1,
      // Upper equatorial band
      4, 2, 5,
      5, 11, 9,
      9, 7, 1,
      1, 6, 8,
      8, 10, 4,
      // Lower equatorial band
      2, 11, 5,
      11, 7, 9,
      7, 6, 1,
      6, 10, 8,
      10, 2, 4,
      // Bottom ring (5 faces)
      3, 11, 2,
      3, 7, 11,
      3, 6, 7,
      3, 10, 6,
      3, 2, 10
    )

    TriangleMeshData(rawVertices, indices, 6)
