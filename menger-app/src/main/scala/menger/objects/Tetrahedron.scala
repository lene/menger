package menger.objects

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import menger.common.TriangleMeshData
import menger.common.TriangleMeshSource

case class Tetrahedron(
  center: Vector3 = Vector3.Zero,
  scale: Float = 1f,
  material: Material = Builder.WHITE_MATERIAL,
  primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(center, scale) with TriangleMeshSource:

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  def getModel: List[ModelInstance] =
    throw UnsupportedOperationException("Tetrahedron is OptiX-only; getModel not supported")

  def toTriangleMesh: TriangleMeshData =
    val cx = center.x
    val cy = center.y
    val cz = center.z

    // Unit tetrahedron vertices on the unit sphere
    val p0x = 0.0f
    val p0y = 1.0f
    val p0z = 0.0f
    val p1x = 0.9428090f
    val p1y = -0.3333333f
    val p1z = 0.0f
    val p2x = -0.4714045f
    val p2y = -0.3333333f
    val p2z = 0.8164966f
    val p3x = -0.4714045f
    val p3y = -0.3333333f
    val p3z = -0.8164966f

    // For a regular tetrahedron inscribed in the unit sphere, vertex position == outward unit normal
    val rawVertices: Array[Float] = Array(
      p0x * scale + cx, p0y * scale + cy, p0z * scale + cz, p0x, p0y, p0z,
      p1x * scale + cx, p1y * scale + cy, p1z * scale + cz, p1x, p1y, p1z,
      p2x * scale + cx, p2y * scale + cy, p2z * scale + cz, p2x, p2y, p2z,
      p3x * scale + cx, p3y * scale + cy, p3z * scale + cz, p3x, p3y, p3z
    )

    // Indices: triangles wound CCW when viewed from outside
    val indices: Array[Int] = Array(
      0, 2, 1,
      0, 3, 2,
      0, 1, 3,
      1, 2, 3
    )

    TriangleMeshData(rawVertices, indices, 6)
