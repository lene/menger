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
    val p0x = 0.0f;       val p0y = 1.0f;         val p0z = 0.0f
    val p1x = 0.9428090f; val p1y = -0.3333333f;   val p1z = 0.0f
    val p2x = -0.4714045f; val p2y = -0.3333333f;  val p2z = 0.8164966f
    val p3x = -0.4714045f; val p3y = -0.3333333f;  val p3z = -0.8164966f

    // Four faces with outward normals: compute face normal as average of the 3 vertices (centroid direction)
    // For a regular tetrahedron inscribed in unit sphere, face normal = normalized centroid of face vertices
    def faceNorm(ax: Float, ay: Float, az: Float, bx: Float, by: Float, bz: Float,
                 ccx: Float, ccy: Float, ccz: Float): (Float, Float, Float) =
      val nx = ax + bx + ccx
      val ny = ay + by + ccy
      val nz = az + bz + ccz
      val len = math.sqrt(nx * nx + ny * ny + nz * nz).toFloat
      (nx / len, ny / len, nz / len)

    val (n0x, n0y, n0z) = faceNorm(p0x, p0y, p0z, p2x, p2y, p2z, p1x, p1y, p1z)
    val (n1x, n1y, n1z) = faceNorm(p0x, p0y, p0z, p3x, p3y, p3z, p2x, p2y, p2z)
    val (n2x, n2y, n2z) = faceNorm(p0x, p0y, p0z, p1x, p1y, p1z, p3x, p3y, p3z)
    val (n3x, n3y, n3z) = faceNorm(p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z)

    // Vertex normals: average of all face normals touching this vertex
    def normalize(x: Float, y: Float, z: Float): (Float, Float, Float) =
      val len = math.sqrt(x * x + y * y + z * z).toFloat
      (x / len, y / len, z / len)

    val (vn0x, vn0y, vn0z) = normalize(n0x + n1x + n2x, n0y + n1y + n2y, n0z + n1z + n2z)
    val (vn1x, vn1y, vn1z) = normalize(n0x + n2x + n3x, n0y + n2y + n3y, n0z + n2z + n3z)
    val (vn2x, vn2y, vn2z) = normalize(n0x + n1x + n3x, n0y + n1y + n3y, n0z + n1z + n3z)
    val (vn3x, vn3y, vn3z) = normalize(n1x + n2x + n3x, n1y + n2y + n3y, n1z + n2z + n3z)

    val rawVertices: Array[Float] = Array(
      p0x * scale + cx, p0y * scale + cy, p0z * scale + cz, vn0x, vn0y, vn0z,
      p1x * scale + cx, p1y * scale + cy, p1z * scale + cz, vn1x, vn1y, vn1z,
      p2x * scale + cx, p2y * scale + cy, p2z * scale + cz, vn2x, vn2y, vn2z,
      p3x * scale + cx, p3y * scale + cy, p3z * scale + cz, vn3x, vn3y, vn3z
    )

    val indices: Array[Int] = Array(
      0, 2, 1,
      0, 3, 2,
      0, 1, 3,
      1, 2, 3
    )

    TriangleMeshData(rawVertices, indices, 6)
