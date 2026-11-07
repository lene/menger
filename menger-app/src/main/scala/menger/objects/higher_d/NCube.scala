package menger.objects.higher_d

import menger.common.Vector

/** The hypercube of dimension `N`: 2^N vertices, N * 2^(N-1) edges and C(N,2) * 2^(N-2) square
  * faces (menger#163).
  *
  * Vertex `i` has coordinate `+size/2` on axis `a` when bit `N-1-a` of `i` is set and `-size/2`
  * otherwise, so axis 0 is the highest bit: for N = 4 the vertex order is [[Tesseract]]'s.
  * A face is one pair of axes (a < b) and a choice of sign on every other axis; its corners run
  * once around the square, a-axis first. `orderedEdges` keeps the order vertex by vertex, axis by
  * axis: anything recorded per edge stays aligned with it. */
final case class NCube[N <: Int & Singleton](size: Float = 1.0f)(using n: ValueOf[N])
    extends Mesh[N]:

  type V = 4

  Dimension.requireSupported(n.value, "n-cube")

  private val dimension: Int = n.value

  /** The bit of a vertex index that selects the sign on `axis`. */
  private def axisBit(axis: Int): Int = 1 << (dimension - 1 - axis)

  private def coordinate(index: Int, axis: Int): Float =
    if (index & axisBit(axis)) != 0 then size / 2 else -size / 2

  lazy val vertices: IndexedSeq[Vector[N]] =
    IndexedSeq.tabulate(NCube.vertexCount(dimension)) { index =>
      Vector[N](IndexedSeq.tabulate(dimension)(coordinate(index, _))*)
    }

  /** Each edge once, as (the vertex with the axis bit unset, the one with it set). */
  lazy val orderedEdges: IndexedSeq[(Vector[N], Vector[N])] =
    for
      index <- vertices.indices
      axis  <- 0 until dimension
      if (index & axisBit(axis)) == 0
    yield (vertices(index), vertices(index | axisBit(axis)))

  lazy val faces: Seq[Polygon[N, V]] =
    for
      a     <- 0 until dimension
      b     <- a + 1 until dimension
      mask   = axisBit(a) | axisBit(b)
      index <- vertices.indices
      if (index & mask) == 0
    yield Polygon[N, V](
      IndexedSeq(index, index | axisBit(a), index | mask, index | axisBit(b)).map(vertices)
    )

object NCube:
  def vertexCount(dimension: Int): Int = 1 << dimension

  def edgeCount(dimension: Int): Int = dimension * (1 << (dimension - 1))

  def faceCount(dimension: Int): Int = dimension * (dimension - 1) / 2 * (1 << (dimension - 2))

  /** The cube of a dimension known only at runtime, drawn through `view`: its vertices are
    * zero-padded to the view's dimension, turned and projected, and a face with a vertex at or
    * behind a stage's eye is left out. */
  def projected(
    dimension: Int, size: Float, view: LiftedProjection, center: Vector[3] = Vector.Zero[3]
  ): ProjectedMesh[?] =
    val d = Dimension.of(dimension)
    given ValueOf[d.N] = d.valueOf
    NCubeProjection[d.N](NCube[d.N](size), view, center)

  /** The cube's edges, in `orderedEdges` order, as 3D endpoints through `view`; None for an
    * edge with an endpoint at or behind a stage's eye. */
  def projectedEdges(
    dimension: Int, size: Float, view: LiftedProjection
  ): IndexedSeq[Option[(Vector[3], Vector[3])]] =
    val d = Dimension.of(dimension)
    given ValueOf[d.N] = d.valueOf
    NCube[d.N](size).orderedEdges.map { (from, to) =>
      for
        p0 <- view.project(from.toIndexedSeq)
        p1 <- view.project(to.toIndexedSeq)
      yield (p0, p1)
    }

/** A [[NCube]] drawn as a triangle mesh through a [[LiftedProjection]]. */
final case class NCubeProjection[N <: Int & Singleton](
  cube: NCube[N],
  view: LiftedProjection,
  center: Vector[3]
)(using ValueOf[N]) extends ProjectedMesh[N]:

  def sourceMesh: Mesh[N] = cube

  override protected def projectVertex(vertex: Vector[N]): Vector[3] = view(vertex.toIndexedSeq)

  override protected def isClipped(vertex: Vector[N]): Boolean =
    view.project(vertex.toIndexedSeq).isEmpty
