package menger.objects.higher_d

import menger.common.Vector

/** `mesh` with every vertex's coordinate on axis k multiplied by `scales(k)`, after the mesh's
  * own size and before rotation and projection (menger#164, from menger#65's w-scale). A factor of
  * 0 collapses the axis: the last factor, 0 -> 1, extrudes an (N-1)-dimensional object along the
  * newest axis into the N-dimensional one. */
final case class AxisScaledMesh[N <: Int & Singleton](mesh: Mesh[N], scales: IndexedSeq[Float])(
  using ValueOf[N]
) extends Mesh[N]:
  type V = mesh.V

  private val scale: Vector.Scale[N] = Vector.Scale[N](scales*)

  def vertices: Seq[Vector[N]] = mesh.vertices.map(scale(_))
  lazy val faces: Seq[Polygon[N, V]] = mesh.faces.map(_.mapVertices(scale(_)))

object AxisScaledMesh:
  /** `scales` unless it is the identity scale, which is no scale at all. */
  def nonIdentity(scales: IndexedSeq[Float]): Option[IndexedSeq[Float]] =
    Some(scales).filterNot(_.forall(_ == 1f))

  /** `mesh` itself at the identity scale. */
  def of[N <: Int & Singleton](mesh: Mesh[N], scales: IndexedSeq[Float])(
    using n: ValueOf[N]
  ): Mesh[N] =
    require(
      scales.size == n.value,
      s"a ${n.value}-dimensional mesh needs ${n.value} scale factors, got ${scales.size}"
    )
    nonIdentity(scales).fold(mesh)(AxisScaledMesh(mesh, _))
