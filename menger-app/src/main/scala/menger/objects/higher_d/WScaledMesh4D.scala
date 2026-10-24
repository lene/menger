package menger.objects.higher_d

import menger.common.Vector

/** `mesh` with every vertex's w multiplied by `wScale` before rotation and projection
  * (menger#65): 0 flattens a tesseract to a cube, animating 0 -> 1 grows it along w -- the
  * extrusion a scene could not express (usability review 2026-10, session 3, F81). */
final case class WScaledMesh4D(mesh: Mesh4D, wScale: Float) extends Mesh4D:
  type V = mesh.V

  private def scaled(v: Vector[4]): Vector[4] = Vector[4](v(0), v(1), v(2), v(3) * wScale)

  def vertices: Seq[Vector[4]] = mesh.vertices.map(scaled)
  lazy val faces: Seq[Face4D[V]] = mesh.faces.map(_.mapVertices(scaled))

object WScaledMesh4D:
  /** `mesh` itself at the identity scale. */
  def of(mesh: Mesh4D, wScale: Float): Mesh4D =
    if wScale == 1f then mesh else WScaledMesh4D(mesh, wScale)
