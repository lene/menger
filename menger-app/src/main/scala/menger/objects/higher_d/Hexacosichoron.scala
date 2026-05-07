package menger.objects.higher_d

import menger.common.Vector

/** Hexacosichoron (600-cell) — dual of the 120-cell.
  * 120 vertices, 720 edges, 1200 triangular faces, 600 tetrahedral cells.
  *
  * Vertex construction on a 4-sphere of circumradius `size`:
  *   - 8  axis vertices: signed unit-axis permutations of (±s, 0, 0, 0)
  *   - 16 hypercube vertices: (±s/2, ±s/2, ±s/2, ±s/2)
  *   - 96 golden vertices: even permutations of (0, ±s/2, ±s/(2φ), ±sφ/2)
  *     where φ = (1+√5)/2 and all 8 sign combinations of the three non-zero values
  *
  * Edge length = s/φ ≈ 0.618·s; edge length² = s²·(2−φ) ≈ 0.382·s². */
case class Hexacosichoron(size: Float = 1f) extends Mesh4D:
  type V = 3

  private val φ    = ((1.0 + math.sqrt(5.0)) / 2.0).toFloat
  private val g1   = size / 2f              // ½s
  private val g2   = size / (2f * φ)       // s/(2φ) = s(φ−1)/2
  private val g3   = size * φ / 2f         // sφ/2

  lazy val vertices: Seq[Vector[4]] =
    // 8 axis: all signed unit-axis vectors (±size, 0, 0, 0)
    val axis = for dim <- 0 until 4; sign <- Seq(-size, size) yield
      val a = Array(0f, 0f, 0f, 0f); a(dim) = sign; Vector[4](a*)

    // 16 hypercube: (±s/2)⁴
    val half = size / 2f
    val hyper = for a <- Seq(-half, half); b <- Seq(-half, half)
                    c <- Seq(-half, half); d <- Seq(-half, half)
      yield Vector[4](a, b, c, d)

    // 96 golden: even permutations of (0, g1, g2, g3) with all sign combinations
    val base = IndexedSeq(0f, g1, g2, g3)
    val evenPerms = IndexedSeq(0,1,2,3).permutations
      .filter(p => inversions(p) % 2 == 0).toSeq
    val golden = for
      perm    <- evenPerms
      bv       = perm.map(base)                  // apply permutation
      nz       = bv.zipWithIndex.filter(_._1 != 0f).map(_._2)
      s0      <- Seq(-1f, 1f); s1 <- Seq(-1f, 1f); s2 <- Seq(-1f, 1f)
    yield
      val arr = bv.toArray
      arr(nz(0)) *= s0; arr(nz(1)) *= s1; arr(nz(2)) *= s2
      Vector[4](arr*)

    (axis ++ hyper ++ golden).distinct

  private def inversions(p: Seq[Int]): Int =
    (for i <- p.indices; j <- i + 1 until p.length if p(i) > p(j) yield 1).sum

  // Squared edge length: s²(2−φ)  ≈ 0.382·s²
  private lazy val edgeLen2: Float = size * size * (2f - φ)
  private lazy val eps2:     Float = edgeLen2 * 0.05f

  // Adjacency by vertex index: used for face and cell construction
  private lazy val adj: Array[Set[Int]] =
    val n = vertices.size
    Array.tabulate(n) { i =>
      (0 until n).filter(j => j != i && {
        val d2 = vertices(i).dst2(vertices(j))
        math.abs(d2 - edgeLen2) < eps2
      }).toSet
    }

  lazy val faces: Seq[Face4D[V]] =
    val n = vertices.size
    val buf = Seq.newBuilder[Face4D[V]]
    for
      i <- 0 until n
      j <- adj(i) if j > i
      k <- adj(i) if k > j && adj(j).contains(k)
    do
      buf += Face4D[3](IndexedSeq(vertices(i), vertices(j), vertices(k)))
    buf.result()

  override lazy val cells: Seq[Cell4D] =
    val n = vertices.size
    val buf = Seq.newBuilder[Cell4D]
    for
      i <- 0 until n
      j <- adj(i) if j > i
      k <- adj(i) if k > j && adj(j).contains(k)
      l <- adj(i) if l > k && adj(j).contains(l) && adj(k).contains(l)
    do
      buf += Seq(vertices(i), vertices(j), vertices(k), vertices(l))
    buf.result()
