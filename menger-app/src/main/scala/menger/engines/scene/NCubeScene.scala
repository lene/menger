package menger.engines.scene

import menger.common.Vector
import menger.common.x
import menger.common.y
import menger.common.z
import menger.failure.Fail
import menger.objects.higher_d.LiftedProjection
import menger.objects.higher_d.NCube
import menger.objects.higher_d.ProjectedMesh
import menger.spec.ObjectSpec
import menger.spec.ProjectionSpec

/** What the engines draw of a `hypercube` spec (menger#163): the n-cube of the spec's
  * dimension seen through its [[ProjectionSpec]] (`ProjectionSpec(dimension)` when it has none),
  * as faces on the CPU path and as one tube per edge. */
object NCubeScene:

  private val MinEdgeLength = 1e-6f

  def dimensionOf(spec: ObjectSpec): Int =
    spec.dimension.getOrElse(Fail.configuration(s"${spec.objectType} needs a dimension"))

  /** The view the cube is seen through; its dimension is at least the cube's. */
  def viewOf(spec: ObjectSpec): ProjectionSpec =
    val view = spec.projectionN.getOrElse(ProjectionSpec(dimensionOf(spec)))
    require(
      view.dimension >= dimensionOf(spec),
      s"a ${dimensionOf(spec)}-cube needs a projection of dimension ${dimensionOf(spec)} or " +
        s"more, got ${view.dimension}"
    )
    view

  private def liftedView(spec: ObjectSpec): LiftedProjection =
    ProjectionConversion.liftedProjection(viewOf(spec))

  def faces(spec: ObjectSpec): ProjectedMesh[?] =
    NCube.projected(dimensionOf(spec), spec.size, liftedView(spec), scales = spec.scale)

  def edgeCount(spec: ObjectSpec): Int = NCube.edgeCount(dimensionOf(spec))

  /** The 3D endpoints of every edge in `NCube.orderedEdges` order, moved to the spec's
    * position; None where an endpoint is at or behind a stage's eye, or where a collapsed axis
    * left the edge without length. */
  def edgeEndpoints(spec: ObjectSpec): IndexedSeq[Option[(Vector[3], Vector[3])]] =
    def placed(p: Vector[3]): Vector[3] = Vector[3](p.x + spec.x, p.y + spec.y, p.z + spec.z)
    NCube.projectedEdges(dimensionOf(spec), spec.size, liftedView(spec), spec.scale).map(
      _.filter((p0, p1) => p0.dst(p1) >= MinEdgeLength).map((p0, p1) => (placed(p0), placed(p1)))
    )
