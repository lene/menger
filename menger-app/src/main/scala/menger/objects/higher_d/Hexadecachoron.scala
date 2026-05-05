package menger.objects.higher_d

import menger.common.Vector

/** Hexadecachoron (16-cell) — 4D orthoplex.
  * 8 vertices: (±s,0,0,0) permutations. 16 tetrahedral cells → 64 triangular faces. */
case class Hexadecachoron(size: Float = 1f) extends Mesh4D:
  type V = 3

  lazy val vertices: Seq[Vector[4]] =
    val s = size * 0.707106781f  // 1/√2
    Seq(
      Vector[4]( s, 0f, 0f, 0f), Vector[4](-s, 0f, 0f, 0f),
      Vector[4](0f,  s, 0f, 0f), Vector[4](0f, -s, 0f, 0f),
      Vector[4](0f, 0f,  s, 0f), Vector[4](0f, 0f, -s, 0f),
      Vector[4](0f, 0f, 0f,  s), Vector[4](0f, 0f, 0f, -s)
    )

  private val cellIndices: Seq[Seq[Int]] = Seq(
    Seq(0,2,4,6), Seq(0,2,4,7), Seq(0,2,5,6), Seq(0,2,5,7),
    Seq(0,3,4,6), Seq(0,3,4,7), Seq(0,3,5,6), Seq(0,3,5,7),
    Seq(1,2,4,6), Seq(1,2,4,7), Seq(1,2,5,6), Seq(1,2,5,7),
    Seq(1,3,4,6), Seq(1,3,4,7), Seq(1,3,5,6), Seq(1,3,5,7)
  )

  override def cells: Seq[Cell4D] = cellIndices.map(c => c.map(vertices))

  lazy val faces: Seq[Face4D[V]] =
    // 16 tetrahedral cells: each cell uses 1 vertex from each opposite pair
    cellIndices.flatMap { cell =>
      val c = cell
      Seq(
        Face4D[3](IndexedSeq(vertices(c(0)), vertices(c(1)), vertices(c(2)))),
        Face4D[3](IndexedSeq(vertices(c(0)), vertices(c(1)), vertices(c(3)))),
        Face4D[3](IndexedSeq(vertices(c(0)), vertices(c(2)), vertices(c(3)))),
        Face4D[3](IndexedSeq(vertices(c(1)), vertices(c(2)), vertices(c(3))))
      )
    }
