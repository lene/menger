package menger.objects

import menger.common.TriangleMeshData
import menger.common.TriangleMeshSource
import menger.common.Vector
import menger.common.x
import menger.common.y
import menger.common.z

case class Tetrahedron(
  center: Vector[3] = Vector.Zero[3],
  scale: Float = 1f
) extends Geometry(center, scale) with TriangleMeshSource:

  def toTriangleMesh: TriangleMeshData =
    val p0 = (0.0f,       1.0f,        0.0f)
    val p1 = (0.9428090f, -0.3333333f,  0.0f)
    val p2 = (-0.4714045f, -0.3333333f, 0.8164966f)
    val p3 = (-0.4714045f, -0.3333333f, -0.8164966f)
    val verts = Array(p0, p1, p2, p3)
    // Faces wound CCW from outside
    val faces = Array((0,2,1), (0,3,2), (0,1,3), (1,2,3))
    PolytopeUtil.flatShaded(faces, verts, scale, center.x, center.y, center.z)
