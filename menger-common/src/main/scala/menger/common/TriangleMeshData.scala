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
