package menger.objects

import com.badlogic.gdx.math.Vector3
import menger.common.TriangleMeshData
import menger.common.TriangleMeshSource

case class Cube(
  center: Vector3 = Vector3.Zero, scale: Float = 1f
) extends Geometry(center, scale) with TriangleMeshSource:

  def toTriangleMesh: TriangleMeshData = toTriangleMeshExcluding(Set.empty)

  def toTriangleMeshExcluding(excluded: Set[Direction]): TriangleMeshData =
    val half = scale / 2
    val faces = Seq(
      Face(center.x + half, center.y, center.z, scale, Direction.X),
      Face(center.x - half, center.y, center.z, scale, Direction.negX),
      Face(center.x, center.y + half, center.z, scale, Direction.Y),
      Face(center.x, center.y - half, center.z, scale, Direction.negY),
      Face(center.x, center.y, center.z + half, scale, Direction.Z),
      Face(center.x, center.y, center.z - half, scale, Direction.negZ)
    ).filterNot(f => excluded.contains(f.normal))
    TriangleMeshData.merge(faces.map(_.toTriangleMesh))
