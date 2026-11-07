package menger.objects.higher_d

import menger.common.Vector

/** A 4D mesh lifted into higher dimensions, turned there and projected to 3D by `view`
  * (menger#162): [[Mesh4DProjection]] with the per-vertex step of a [[LiftedProjection]]. */
final case class LiftedMesh4DProjection(
  mesh4D: Mesh4D,
  view: LiftedProjection,
  center: Vector[3] = Vector[3](0f, 0f, 0f)
) extends ProjectedMesh4D:

  override protected def projectVertex(vertex: Vector[4]): Vector[3] = view(vertex)
