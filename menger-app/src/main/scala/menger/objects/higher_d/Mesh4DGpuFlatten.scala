package menger.objects.higher_d

/** Sprint 18.3 Cut B: turn a `Mesh4D` (sequence of Face4D faces) into the flat
  * float buffer expected by `OptiXRenderer.setProjectedMesh`.
  *
  * Layout: N faces × V corners × (x, y, z, w), corners ordered sequentially.
  */
object Mesh4DGpuFlatten:

  def facesBuffer(mesh4D: Mesh4D): (Array[Float], Int) =
    val vpf = mesh4D.vertsPerFace
    val buffer = mesh4D.faces.iterator.flatMap { f =>
      (0 until vpf).iterator.flatMap { i =>
        val v = f(i)
        Iterator(v(0), v(1), v(2), v(3))
      }
    }.toArray
    (buffer, vpf)

  @SuppressWarnings(Array("org.wartremover.warts.AsInstanceOf"))
  def quadsBuffer(mesh4D: Mesh4D, normalOffset: Float = 0f): Array[Float] =
    require(mesh4D.vertsPerFace == 4, "quadsBuffer requires quad faces (vertsPerFace=4)")
    mesh4D.faces.asInstanceOf[Seq[Face4D[4]]].iterator.flatMap { f =>
      if normalOffset == 0f then
        vertexFloats(f(0)) ++ vertexFloats(f(1)) ++ vertexFloats(f(2)) ++ vertexFloats(f(3))
      else
        val offset = f.normals.foldLeft(menger.common.Vector.Zero[4])(_ + _) * normalOffset
        vertexFloats(f(0) + offset) ++ vertexFloats(f(1) + offset) ++
          vertexFloats(f(2) + offset) ++ vertexFloats(f(3) + offset)
    }.toArray

  private inline def vertexFloats(v: menger.common.Vector[4]): Array[Float] =
    Array(v(0), v(1), v(2), v(3))
