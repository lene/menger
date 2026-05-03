package menger.objects

import menger.common.TriangleMeshData

/**
 * Shared utilities for platonic solid geometry construction.
 */
object PolytopeUtil:

  /**
   * Converts indexed triangle faces into a flat-shaded TriangleMeshData.
   *
   * Each triangle is expanded to 3 dedicated vertices all carrying the same
   * geometric face normal (computed via cross product). This prevents the OptiX
   * shader from interpolating normals between adjacent faces, which would produce
   * a curved-surface appearance inappropriate for polyhedra.
   *
   * @param faces  Triangle faces as (i0, i1, i2) index triples, CCW from outside
   * @param verts  Vertex positions on the unit sphere (before scale/translate)
   * @param scale  Uniform scale factor applied to positions
   * @param cx     X component of center translation
   * @param cy     Y component of center translation
   * @param cz     Z component of center translation
   */
  def flatShaded(
    faces: Array[(Int, Int, Int)],
    verts: Array[(Float, Float, Float)],
    scale: Float,
    cx: Float, cy: Float, cz: Float
  ): TriangleMeshData =
    val rawVertices: Array[Float] = faces.flatMap { (i0, i1, i2) =>
      val (ax, ay, az) = verts(i0)
      val (bx, by, bz) = verts(i1)
      val (dx, dy, dz) = verts(i2)
      // Geometric face normal via cross product of edge vectors (e1 × e2)
      val e1x = bx - ax; val e1y = by - ay; val e1z = bz - az
      val e2x = dx - ax; val e2y = dy - ay; val e2z = dz - az
      val nx = e1y * e2z - e1z * e2y
      val ny = e1z * e2x - e1x * e2z
      val nz = e1x * e2y - e1y * e2x
      val nlen = math.sqrt(nx * nx + ny * ny + nz * nz).toFloat
      val nnx = nx / nlen
      val nny = ny / nlen
      val nnz = nz / nlen
      Array(
        ax * scale + cx, ay * scale + cy, az * scale + cz, nnx, nny, nnz,
        bx * scale + cx, by * scale + cy, bz * scale + cz, nnx, nny, nnz,
        dx * scale + cx, dy * scale + cy, dz * scale + cz, nnx, nny, nnz
      )
    }
    // Sequential indices: vertex i belongs to triangle i/3, no sharing
    val indices: Array[Int] = Array.range(0, rawVertices.length / 6)
    TriangleMeshData(rawVertices, indices, 6)
