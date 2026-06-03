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
  def quadsBuffer(mesh4D: Mesh4D, skinOffset: Float = 0f): Array[Float] =
    require(mesh4D.vertsPerFace == 4, "quadsBuffer requires quad faces (vertsPerFace=4)")
    // A non-zero skinOffset expands the lower-level mesh of a fractional pair so it
    // does not z-fight the coincident higher-level surface. We use a uniform radial
    // scale about the (origin-centred) sponge rather than a per-face normal offset:
    // the per-face approach (Σ winding-signed normals) moved faces in inconsistent
    // directions and split shared vertices, opening gaps that showed the surface
    // behind as dark squares at the cube corners. A radial scale moves coincident
    // vertices identically, so it is gap-free by construction.
    val scale = 1f + skinOffset
    mesh4D.faces.asInstanceOf[Seq[Face4D[4]]].iterator.flatMap { f =>
      if skinOffset == 0f then
        vertexFloats(f(0)) ++ vertexFloats(f(1)) ++ vertexFloats(f(2)) ++ vertexFloats(f(3))
      else
        vertexFloats(f(0) * scale) ++ vertexFloats(f(1) * scale) ++
          vertexFloats(f(2) * scale) ++ vertexFloats(f(3) * scale)
    }.toArray

  private inline def vertexFloats(v: menger.common.Vector[4]): Array[Float] =
    Array(v(0), v(1), v(2), v(3))
