package menger.objects.higher_d

import menger.common.Vector

/** Icositetrachoron (24-cell) — self-dual 4D polytope.
  * 24 vertices, 24 octahedral cells → 192 triangular Face4D[3] faces. */
case class Icositetrachoron(size: Float = 1f) extends Mesh4D:
  type V = 3

  private val s = size * 0.5f
  private val h = s * 0.5f

  // 8 axial: all signed permutations of (s, 0, 0, 0)
  private val axialVertices: IndexedSeq[Vector[4]] =
    (for dim <- 0 until 4; sign <- Seq(-s, s) yield
      val a = Array(0f, 0f, 0f, 0f)
      a(dim) = sign
      Vector[4](a*)
    ).toIndexedSeq

  // 16 half: all signed permutations of (h, h, h, h)
  private val halfVertices: IndexedSeq[Vector[4]] =
    (for a <- Seq(-h, h); b <- Seq(-h, h); c <- Seq(-h, h); d <- Seq(-h, h)
      yield Vector[4](a, b, c, d)
    ).toIndexedSeq

  lazy val vertices: Seq[Vector[4]] = axialVertices ++ halfVertices

  // Bit index for half vertex: c0 is outermost loop (weight 8), c3 innermost (weight 1)
  private def halfIdx(c0: Float, c1: Float, c2: Float, c3: Float): Int =
    8 + ((if c0 > 0f then 8 else 0) |
         (if c1 > 0f then 4 else 0) |
         (if c2 > 0f then 2 else 0) |
         (if c3 > 0f then 1 else 0))

  override def cells: Seq[Cell4D] =
    for
      i  <- 0 until 4
      j  <- (i + 1) until 4
      si <- Seq(-1, 1)
      sj <- Seq(-1, 1)
    yield
      val pole0 = axialVertices(i * 2 + (if si > 0 then 1 else 0))
      val pole1 = axialVertices(j * 2 + (if sj > 0 then 1 else 0))
      val free  = (0 until 4).filter(k => k != i && k != j).toIndexedSeq
      val d0 = free(0); val d1 = free(1)
      val ring =
        for s0 <- Seq(-h, h); s1 <- Seq(-h, h) yield
          val a = Array(0f, 0f, 0f, 0f)
          a(i) = si * h; a(j) = sj * h; a(d0) = s0; a(d1) = s1
          Vector[4](a*)
      Seq(pole0, pole1) ++ ring

  lazy val faces: Seq[Face4D[V]] =
    val builder = Seq.newBuilder[Face4D[V]]
    for
      i <- 0 until 4
      j <- 0 until 4
      if i != j
      si <- Seq(-1, 1)
      sj <- Seq(-1, 1)
    do
      val pole0Idx = i * 2 + (if si > 0 then 1 else 0)
      val pole1Idx = j * 2 + (if sj > 0 then 1 else 0)
      val free = (0 until 4).filter(k => k != i && k != j).toIndexedSeq
      val d0 = free(0)
      val d1 = free(1)
      def makeCoords(s0: Float, s1: Float): (Float, Float, Float, Float) =
        val a = Array(0f, 0f, 0f, 0f)
        a(i) = si * h; a(j) = sj * h; a(d0) = s0; a(d1) = s1
        (a(0), a(1), a(2), a(3))
      val (c0a, c0b, c0c, c0d) = makeCoords(h, h)
      val (c1a, c1b, c1c, c1d) = makeCoords(-h, h)
      val (c2a, c2b, c2c, c2d) = makeCoords(-h, -h)
      val (c3a, c3b, c3c, c3d) = makeCoords(h, -h)
      val e0 = halfIdx(c0a, c0b, c0c, c0d)
      val e1 = halfIdx(c1a, c1b, c1c, c1d)
      val e2 = halfIdx(c2a, c2b, c2c, c2d)
      val e3 = halfIdx(c3a, c3b, c3c, c3d)
      val p0 = vertices(pole0Idx)
      val p1 = vertices(pole1Idx)
      val v0 = vertices(e0); val v1 = vertices(e1)
      val v2 = vertices(e2); val v3 = vertices(e3)
      builder += Face4D[3](IndexedSeq(p0, v0, v1))
      builder += Face4D[3](IndexedSeq(p0, v1, v2))
      builder += Face4D[3](IndexedSeq(p0, v2, v3))
      builder += Face4D[3](IndexedSeq(p0, v3, v0))
      builder += Face4D[3](IndexedSeq(p1, v1, v0))
      builder += Face4D[3](IndexedSeq(p1, v2, v1))
      builder += Face4D[3](IndexedSeq(p1, v3, v2))
      builder += Face4D[3](IndexedSeq(p1, v0, v3))
    builder.result().distinctBy(f => f.asSeq.map(v => (v(0), v(1), v(2), v(3))).toSet)
