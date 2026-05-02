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

  /** Flatten a `Mesh4D` to a `Float[N*16]` quads buffer.
    *
    * When `normalOffset != 0`, each face's vertices are shifted by
    * `offset * sumOfFaceNormals` before flattening — equivalent to the CPU
    * path's `expandAlongNormals`, preventing z-fighting when two meshes share
    * the same 4D surface (e.g. fractional sponge level n vs level n+1).
    * Requires axis-aligned quad faces (always true for sponge geometry). */
  def quadsBuffer(mesh4D: Mesh4D, normalOffset: Float = 0f): Array[Float] =
    mesh4D.faces.iterator.flatMap { f =>
      if normalOffset == 0f then
        vertexFloats(f.a) ++ vertexFloats(f.b) ++ vertexFloats(f.c) ++ vertexFloats(f.d)
      else
        val offset = f.normals.foldLeft(menger.common.Vector.Zero[4])(_ + _) * normalOffset
        vertexFloats(f.a + offset) ++ vertexFloats(f.b + offset) ++
          vertexFloats(f.c + offset) ++ vertexFloats(f.d + offset)
    }.toArray

  private inline def vertexFloats(v: menger.common.Vector[4]): Array[Float] =
    Array(v(0), v(1), v(2), v(3))
