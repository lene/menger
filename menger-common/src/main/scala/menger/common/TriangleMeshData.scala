package menger.common

/** Renderer-agnostic triangle mesh data.
  *
  * Contains interleaved vertex data and triangle indices. Can be consumed by both LibGDX and OptiX
  * renderers.
  *
  * Vertex formats supported:
  *   - 6 floats per vertex: position(3) + normal(3)
  *   - 8 floats per vertex: position(3) + normal(3) + uv(2) [default, per AD-4]
  *
  * Index format: 3 indices per triangle
  */
case class TriangleMeshData(
    vertices: Array[Float],
    indices: Array[Int],
    vertexStride: Int = TriangleMeshData.DefaultVertexStride
):
  require(
    vertexStride == 6 || vertexStride == 8 || vertexStride == 9,
    s"Vertex stride must be 6 (pos+normal), 8 (pos+normal+uv), or 9 (pos+normal+uv+alpha), got $vertexStride"
  )
  require(
    vertices.isEmpty || vertices.length % vertexStride == 0,
    s"Vertices must have $vertexStride floats per vertex, got ${vertices.length}"
  )
  require(
    indices.length % 3 == 0,
    s"Indices must have 3 per triangle, got ${indices.length}"
  )

  def numVertices: Int = if vertices.isEmpty then 0 else vertices.length / vertexStride
  def numTriangles: Int = indices.length / 3
  def hasUVs: Boolean = vertexStride == 8

object TriangleMeshData:
  /** Default vertex stride includes UV coordinates (position + normal + UV = 8 floats) */
  val DefaultVertexStride: Int = 8

  /** Legacy vertex stride without UV coordinates (position + normal = 6 floats) */
  val LegacyVertexStride: Int = 6

  /** Extended vertex stride with per-vertex alpha (position + normal + UV + alpha = 9 floats) */
  val VertexStrideWithAlpha: Int = 9

  val empty: TriangleMeshData = TriangleMeshData(Array.emptyFloatArray, Array.emptyIntArray, 6)

  def merge(meshes: Seq[TriangleMeshData]): TriangleMeshData =
    if meshes.isEmpty then empty
    else if meshes.size == 1 then meshes.head
    else
      val stride = meshes.head.vertexStride
      require(
        meshes.forall(_.vertexStride == stride),
        s"Cannot merge meshes with different vertex strides: ${meshes.map(_.vertexStride).distinct}"
      )
      val allVertices = meshes.flatMap(_.vertices).toArray
      val allIndices = adjustIndices(meshes)
      TriangleMeshData(allVertices, allIndices, stride)

  private def adjustIndices(meshes: Seq[TriangleMeshData]): Array[Int] =
    val (indices, _) = meshes.foldLeft((Array.emptyIntArray, 0)) { case ((acc, offset), mesh) =>
      val adjusted = mesh.indices.map(_ + offset)
      (acc ++ adjusted, offset + mesh.numVertices)
    }
    indices

  /**
   * Expand all vertices outward along their normals by a fixed world-space offset.
   *
   * Used for fractional sponge skin meshes to prevent z-fighting with the underlying
   * sponge faces. Moves each vertex by offset * normal, pushing the face slightly outward
   * (toward the viewer) so it sits in front of the sponge face behind it.
   *
   * @param mesh   Source mesh (any supported stride)
   * @param offset World-space offset to expand along normals (positive = outward)
   * @return New mesh with expanded vertex positions, same stride and indices
   */
  def expandAlongNormals(mesh: TriangleMeshData, offset: Float): TriangleMeshData =
    require(mesh.vertexStride >= 6, s"Mesh must have at least stride 6, got ${mesh.vertexStride}")
    val newVertices = mesh.vertices.clone()
    for (i <- 0 until mesh.numVertices) {
      val base = i * mesh.vertexStride
      // Position offsets: px += offset * nx, py += offset * ny, pz += offset * nz
      newVertices(base)     += offset * mesh.vertices(base + 3)
      newVertices(base + 1) += offset * mesh.vertices(base + 4)
      newVertices(base + 2) += offset * mesh.vertices(base + 5)
    }
    TriangleMeshData(newVertices, mesh.indices, mesh.vertexStride)

  /**
   * Add per-vertex alpha channel to mesh, extending stride from 8 to 9.
   *
   * Takes a mesh with stride=8 (pos+normal+uv) and returns a new mesh with stride=9
   * (pos+normal+uv+alpha), where all vertices are assigned the given alpha value.
   *
   * This is used for fractional level rendering where different parts of the merged
   * geometry need different transparency levels.
   *
   * @param mesh Source mesh with stride=8
   * @param alpha Alpha value to assign to all vertices (0.0 = transparent, 1.0 = opaque)
   * @return New mesh with stride=9 and alpha channel added
   */
  def withAlpha(mesh: TriangleMeshData, alpha: Float): TriangleMeshData =
    require(mesh.vertexStride == 8, s"Can only add alpha to stride=8 meshes, got stride=${mesh.vertexStride}")
    require(alpha >= 0.0f && alpha <= 1.0f, s"Alpha must be in [0.0, 1.0], got $alpha")

    val newVertices = new Array[Float](mesh.numVertices * 9)
    for (i <- 0 until mesh.numVertices) {
      val srcOffset = i * 8
      val dstOffset = i * 9

      // Copy pos + normal + uv (8 floats)
      System.arraycopy(mesh.vertices, srcOffset, newVertices, dstOffset, 8)

      // Add alpha as 9th component
      newVertices(dstOffset + 8) = alpha
    }

    TriangleMeshData(newVertices, mesh.indices, vertexStride = 9)
