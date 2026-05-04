package menger.objects.higher_d

import menger.common.Vector

/** Regular pentachoron (5-cell) — 4D simplex.
  * 5 vertices, 10 edges, 10 triangular faces (2D), 5 tetrahedral cells (3D).
  * Decomposes into 20 triangular Face4D[3] faces for rendering. */
case class Pentachoron(size: Float = 1f) extends Mesh4D:
  type V = 3

  lazy val vertices: Seq[Vector[4]] =
    val s = size * 0.5f
    // Regular simplex coordinates (unit edge length, centered at origin)
    val r5 = scala.math.sqrt(0.4).toFloat * s     // √(2/5)
    val r0 = s                                     // 1
    Seq(
      Vector[4]( r0,  r0,  r0, -r5),
      Vector[4]( r0, -r0, -r0, -r5),
      Vector[4](-r0,  r0, -r0, -r5),
      Vector[4](-r0, -r0,  r0, -r5),
      Vector[4]( 0f,  0f,  0f, 4f * r5)
    )

  lazy val faces: Seq[Face4D[V]] =
    // 5 tetrahedral cells, each with 4 triangular faces
    // Cell vertex indices: (0,1,2,4), (0,1,3,4), (0,2,3,4), (1,2,3,4), (0,1,2,3)
    val cells = Seq(
      Seq(0, 1, 2, 4), Seq(0, 1, 3, 4), Seq(0, 2, 3, 4),
      Seq(1, 2, 3, 4), Seq(0, 1, 2, 3)
    )
    cells.flatMap { cell =>
      // Each tetrahedron has 4 triangular faces
      val c = cell
      Seq(
        Face4D[3](IndexedSeq(vertices(c(0)), vertices(c(1)), vertices(c(2)))),
        Face4D[3](IndexedSeq(vertices(c(0)), vertices(c(1)), vertices(c(3)))),
        Face4D[3](IndexedSeq(vertices(c(0)), vertices(c(2)), vertices(c(3)))),
        Face4D[3](IndexedSeq(vertices(c(1)), vertices(c(2)), vertices(c(3))))
      )
    }
