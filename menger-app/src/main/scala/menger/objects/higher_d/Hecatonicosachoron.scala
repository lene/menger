package menger.objects.higher_d

import menger.common.Vector

/** Hecatonicosachoron (120-cell) — dual of the 600-cell.
  * 600 vertices, 1200 edges, 720 pentagonal faces, 120 dodecahedral cells.
  *
  * Constructed as the centroid dual of the 600-cell (Hexacosichoron):
  *   - each tetrahedral cell  → one vertex  (its centroid)
  *   - each triangular face   → one edge
  *   - each edge              → one pentagonal face (5 cells around each edge)
  *   - each vertex            → one dodecahedral cell (20 cells around each vertex)
  *
  * Edge length = size/φ² ≈ 0.382·size; circumradius = size. */
case class Hecatonicosachoron(size: Float = 1f) extends Mesh4D:
  type V = 5

  private val h        = Hexacosichoron(1f)
  private val cells600 = h.cells.toIndexedSeq    // 600 tetrahedral cells
  private val verts600 = h.vertices.toIndexedSeq // 120 vertices

  // ─── raw vertices = centroids of 600-cell cells ───────────────────────────

  private def centroid(cell: Seq[Vector[4]]): Vector[4] =
    cell.reduce(_ + _) / 4f

  private lazy val rawVerts: IndexedSeq[Vector[4]] = cells600.map(centroid)

  private lazy val rawR: Float = math.sqrt(rawVerts.head.len2).toFloat

  lazy val vertices: Seq[Vector[4]] = rawVerts.map(_ * (size / rawR))

  // ─── 600-cell index structures ────────────────────────────────────────────

  private lazy val v600Idx: Map[Vector[4], Int] =
    verts600.zipWithIndex.toMap

  private lazy val cellVerts: IndexedSeq[Set[Int]] =
    cells600.map(cell => cell.map(v => v600Idx(v)).toSet)

  private lazy val vertToCells: Map[Int, Set[Int]] =
    cellVerts.zipWithIndex
      .flatMap { case (vset, ci) => vset.map(vi => vi -> ci) }
      .groupBy(_._1)
      .view.mapValues(_.map(_._2).toSet)
      .toMap

  // ─── 600-cell edges (from its triangular faces) ───────────────────────────

  private lazy val edges600: Seq[(Int, Int)] =
    h.faces.flatMap { f =>
      val vs = (0 until 3).map(i => v600Idx(f(i)))
      Seq(
        (vs(0) min vs(1), vs(0) max vs(1)),
        (vs(1) min vs(2), vs(1) max vs(2)),
        (vs(0) min vs(2), vs(0) max vs(2))
      )
    }.distinct

  // ─── face construction ────────────────────────────────────────────────────

  // Order the 5 cells into a pentagon ring around edge (uIdx, vIdx).
  // Consecutive cells share a 600-cell triangular face containing that edge.
  private def orderRing(cells5: IndexedSeq[Int], uIdx: Int, vIdx: Int): IndexedSeq[Int] =
    def others(ci: Int): Set[Int] = cellVerts(ci) - uIdx - vIdx

    def ringNeighbors(ci: Int): Set[Int] =
      cells5.filter(cj => cj != ci && (cellVerts(cj) & others(ci)).size == 1).toSet

    @scala.annotation.tailrec
    def go(cur: Int, prev: Int, acc: IndexedSeq[Int]): IndexedSeq[Int] =
      if acc.size == cells5.size then acc
      else
        val next = (ringNeighbors(cur) - prev).head
        go(next, cur, acc :+ next)

    val first  = cells5.head
    val second = ringNeighbors(first).head
    go(second, first, IndexedSeq(first, second))

  lazy val faces: Seq[Face4D[V]] =
    edges600.flatMap { case (uIdx, vIdx) =>
      val cells5 = (vertToCells(uIdx) & vertToCells(vIdx)).toIndexedSeq
      Option.when(cells5.size == 5) {
        val ring = orderRing(cells5, uIdx, vIdx)
        Face4D[5](ring.map(ci => vertices(ci)))
      }
    }

  override lazy val cells: Seq[Cell4D] =
    (0 until verts600.size).map { vi =>
      vertToCells(vi).toSeq.map(ci => vertices(ci))
    }
