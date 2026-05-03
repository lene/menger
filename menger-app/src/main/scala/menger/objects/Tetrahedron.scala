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
    val p0 = (0.0f,       1.0f,        0.0f)
    val p1 = (0.9428090f, -0.3333333f,  0.0f)
    val p2 = (-0.4714045f, -0.3333333f, 0.8164966f)
    val p3 = (-0.4714045f, -0.3333333f, -0.8164966f)
    val verts = Array(p0, p1, p2, p3)
    // Faces wound CCW from outside
    val faces = Array((0,2,1), (0,3,2), (0,1,3), (1,2,3))
    PolytopeUtil.flatShaded(faces, verts, scale, center.x, center.y, center.z)
