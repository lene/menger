package menger.common

/** Renderer-agnostic triangle mesh data.
  *
  * Contains interleaved vertex data (position + normal) and triangle indices. Can be consumed by
  * both LibGDX and OptiX renderers.
  *
  * Vertex format: 6 floats per vertex (px, py, pz, nx, ny, nz) Index format: 3 indices per
  * triangle
  */
case class TriangleMeshData(
    vertices: Array[Float],
    indices: Array[Int]
):
  require(
    vertices.length % 6 == 0,
    s"Vertices must have 6 floats per vertex (pos + normal), got ${vertices.length}"
  )
  require(
    indices.length % 3 == 0,
    s"Indices must have 3 per triangle, got ${indices.length}"
  )

  def numVertices: Int = vertices.length / 6
  def numTriangles: Int = indices.length / 3

object TriangleMeshData:

  val empty: TriangleMeshData = TriangleMeshData(Array.emptyFloatArray, Array.emptyIntArray)

  def merge(meshes: Seq[TriangleMeshData]): TriangleMeshData =
    if meshes.isEmpty then empty
    else if meshes.size == 1 then meshes.head
    else
      val allVertices = meshes.flatMap(_.vertices).toArray
      val allIndices = adjustIndices(meshes)
      TriangleMeshData(allVertices, allIndices)

  private def adjustIndices(meshes: Seq[TriangleMeshData]): Array[Int] =
    val (indices, _) = meshes.foldLeft((Array.emptyIntArray, 0)) { case ((acc, offset), mesh) =>
      val adjusted = mesh.indices.map(_ + offset)
      (acc ++ adjusted, offset + mesh.numVertices)
    }
    indices
