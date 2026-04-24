package menger.objects.higher_d

import menger.common.TriangleMeshData

/** Topology checker for mesh manifoldness.
  *
  * Two separate functions — one for 4D quad faces (pre-projection), one for
  * 3D triangle soups (post-projection) — because the two stages have different
  * vertex-equality semantics: 4D vertices are built from the same rational
  * arithmetic so exact float equality holds, while 3D vertices after perspective
  * projection may differ by float rounding at shared seams.
  */
object MeshTopology:

  /** Summary of an edge-sharing check. */
  case class TopologyReport(
      faceCount: Int,
      edgeUseHistogram: Map[Int, Int],
      boundaryFaces: Seq[Int]
  ):
    def isManifold: Boolean = edgeUseHistogram.keys.forall(_ == 2)
    def boundaryEdgeCount: Int = edgeUseHistogram.getOrElse(1, 0)

  /** Check manifoldness of a Seq[Face4D] at the 4D level.
    *
    * Edges are keyed by exact float coordinates — valid because 4D vertices are
    * computed from the same rational-fraction arithmetic so shared vertices are
    * bit-identical. Returns boundary face indices (faces with >= 1 edge used
    * by only one face).
    */
  def checkFace4D(faces: Seq[Face4D]): TopologyReport =
    type VKey = (Float, Float, Float, Float)
    type EKey = (VKey, VKey)

    def vkey(v: menger.common.Vector[4]): VKey = (v(0), v(1), v(2), v(3))
    def ekey(a: VKey, b: VKey): EKey =
      if summon[Ordering[VKey]].lteq(a, b) then (a, b) else (b, a)

    val faceEdges: Seq[(EKey, Int)] =
      faces.zipWithIndex.flatMap { case (f, fi) =>
        f.edges.map(e => ekey(vkey(e.v0), vkey(e.v1)) -> fi)
      }

    buildReport(faces.size, faceEdges)

  /** Check manifoldness of a TriangleMeshData at the 3D level.
    *
    * Two vertices are considered the same edge endpoint if their XYZ positions
    * differ by at most edgeEpsilon in each coordinate. Default 1e-4f is well
    * above floating-point rounding for unit-scale geometry but tight enough to
    * distinguish neighbouring vertices.
    */
  def checkTriangleMesh(mesh: TriangleMeshData, edgeEpsilon: Float = 1e-4f): TopologyReport =
    type PKey = (Long, Long, Long)
    type EKey = (PKey, PKey)

    def pkey(vi: Int): PKey =
      val base = vi * mesh.vertexStride
      ((mesh.vertices(base)     / edgeEpsilon).round,
       (mesh.vertices(base + 1) / edgeEpsilon).round,
       (mesh.vertices(base + 2) / edgeEpsilon).round)

    def ekey(a: PKey, b: PKey): EKey =
      if summon[Ordering[PKey]].lteq(a, b) then (a, b) else (b, a)

    val triEdges: Seq[(EKey, Int)] =
      (0 until mesh.numTriangles).flatMap { ti =>
        val i0 = mesh.indices(3 * ti)
        val i1 = mesh.indices(3 * ti + 1)
        val i2 = mesh.indices(3 * ti + 2)
        Seq(
          ekey(pkey(i0), pkey(i1)) -> ti,
          ekey(pkey(i1), pkey(i2)) -> ti,
          ekey(pkey(i2), pkey(i0)) -> ti
        )
      }

    buildReport(mesh.numTriangles, triEdges)

  private def buildReport[K](faceCount: Int, faceEdges: Seq[(K, Int)]): TopologyReport =
    val edgeToFaces: Map[K, Seq[Int]] = faceEdges.groupMap(_._1)(_._2)

    val histogram: Map[Int, Int] =
      edgeToFaces.values.groupBy(_.size).view.mapValues(_.size).toMap

    val boundary: Seq[Int] =
      edgeToFaces.collect { case (_, fis) if fis.size == 1 => fis.head }
        .toSet.toSeq.sorted

    TopologyReport(faceCount, histogram, boundary)
