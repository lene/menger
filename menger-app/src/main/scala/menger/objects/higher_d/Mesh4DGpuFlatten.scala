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
  def quadsBuffer(mesh4D: Mesh4D): Array[Float] =
    require(mesh4D.vertsPerFace == 4, "quadsBuffer requires quad faces (vertsPerFace=4)")
    mesh4D.faces.asInstanceOf[Seq[Face4D[4]]].iterator.flatMap { f =>
      vertexFloats(f(0)) ++ vertexFloats(f(1)) ++ vertexFloats(f(2)) ++ vertexFloats(f(3))
    }.toArray

  private val CapFraction = 1f / 3f

  /** The hole caps of a fractional 4D sponge's lower level, in `quadsBuffer` layout: each face
    * shrunk to its centre third about its own centroid, which is exactly the opening the next
    * level leaves in that face. Replaces uploading the whole lower level, radially scaled by
    * 1.0003, whose near-coincident surface caused speckle, double refraction and shifted
    * procedural colours (usability review 2026-09, session 2, F35/F41). */
  @SuppressWarnings(Array("org.wartremover.warts.AsInstanceOf"))
  def holeCapsBuffer(mesh4D: Mesh4D): Array[Float] =
    require(mesh4D.vertsPerFace == 4, "holeCapsBuffer requires quad faces (vertsPerFace=4)")
    mesh4D.faces.asInstanceOf[Seq[Face4D[4]]].iterator.flatMap { f =>
      val corners = (0 until 4).map(f(_))
      val centre = corners.reduce(_ + _) / 4f
      corners.flatMap(v => vertexFloats(centre + (v - centre) * CapFraction))
    }.toArray

  private inline def vertexFloats(v: menger.common.Vector[4]): Array[Float] =
    Array(v(0), v(1), v(2), v(3))
