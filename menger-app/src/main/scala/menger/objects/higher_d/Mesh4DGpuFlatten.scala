package menger.objects.higher_d

/** Sprint 18.3 Cut B: turn a `Mesh4D` (sequence of Face4D quads) into the flat
  * `Float[N*16]` buffer expected by `OptiXRenderer.setTriangleMesh4DQuads`.
  *
  * Layout: N quads × 4 corners × (x, y, z, w), corners ordered (a, b, c, d)
  * to match `Face4D.asSeq`. UVs are not produced here — the kernel falls back
  * to default unit-square UVs which match `Mesh4DProjection.quadToTriangleMesh`.
  *
  * No projection logic — that now runs on the GPU. This utility purely
  * flattens the 4D vertex array.
  */
object Mesh4DGpuFlatten:

  def quadsBuffer(mesh4D: Mesh4D): Array[Float] =
    mesh4D.faces.iterator.flatMap { f =>
      vertexFloats(f.a) ++ vertexFloats(f.b) ++ vertexFloats(f.c) ++ vertexFloats(f.d)
    }.toArray

  private inline def vertexFloats(v: menger.common.Vector[4]): Array[Float] =
    Array(v(0), v(1), v(2), v(3))
