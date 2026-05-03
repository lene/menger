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
    val verts = Array(
      (1f, 0f, 0f), (-1f, 0f, 0f), (0f, 1f, 0f),
      (0f, -1f, 0f), (0f, 0f, 1f), (0f, 0f, -1f)
    )
    // 8 triangular faces wound CCW from outside
    val faces = Array(
      (2,4,0), (2,1,4), (2,5,1), (2,0,5),
      (3,0,4), (3,4,1), (3,1,5), (3,5,0)
    )
    PolytopeUtil.flatShaded(faces, verts, scale, center.x, center.y, center.z)
