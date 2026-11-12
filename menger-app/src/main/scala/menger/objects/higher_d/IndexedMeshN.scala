package menger.objects.higher_d

import menger.common.TriangleMeshData
import menger.common.Vector

/** Square faces in `dimension` dimensions as an indexed mesh (menger#166): `dimension` floats
  * per vertex and four vertex indices per face, in cyclic order around the square. A shared
  * vertex is stored, and projected, once however many faces use it. */
final case class IndexedMeshN(dimension: Int, vertices: Array[Float], quads: Array[Int]):
  require(dimension > 0, s"dimension must be positive, got $dimension")
  require(
    vertices.length % dimension == 0,
    s"vertices must hold $dimension floats each, got ${vertices.length}"
  )
  require(
    quads.length % IndexedMeshN.CornersPerFace == 0,
    s"faces must have ${IndexedMeshN.CornersPerFace} indices each, got ${quads.length}"
  )

  def vertexCount: Int = vertices.length / dimension

  def faceCount: Int = quads.length / IndexedMeshN.CornersPerFace

  def faceVertices(face: Int): IndexedSeq[IndexedSeq[Float]] =
    cornerIndices(face).map(vertex =>
      IndexedSeq.tabulate(dimension)(axis => vertices(vertex * dimension + axis))
    )

  /** Every face replaced by the square 1/3 its size around the same centroid: the cap of a
    * hole in a face split into nine. The caps share no vertices. */
  def holeCaps: IndexedMeshN =
    val capVertices = Array.tabulate(faceCount * IndexedMeshN.CornersPerFace * dimension) { slot =>
      val face = slot / (IndexedMeshN.CornersPerFace * dimension)
      val corner = slot / dimension % IndexedMeshN.CornersPerFace
      val axis = slot % dimension
      val centroid = coordinateSum(face, axis) / IndexedMeshN.CornersPerFace
      val vertex = quads(face * IndexedMeshN.CornersPerFace + corner)
      val coordinate = vertices(vertex * dimension + axis)
      centroid + (coordinate - centroid) / IndexedMeshN.CapDivisor
    }
    IndexedMeshN(
      dimension,
      capVertices,
      Array.tabulate(faceCount * IndexedMeshN.CornersPerFace)(identity)
    )

  /** The faces as triangles: each vertex is scaled by `size` and then axis by axis by `scales`,
    * projected once, and a face with a vertex that `project` rejects is dropped. The triangles
    * are those [[ProjectedMesh]] makes of the same faces. */
  def toTriangleMesh(
    size: Float,
    scales: IndexedSeq[Float],
    project: IndexedSeq[Float] => Option[Vector[3]],
    center: Vector[3]
  ): TriangleMeshData =
    require(
      scales.size == dimension,
      s"need one scale per axis: $dimension axes, got ${scales.size} scales"
    )
    val projected: Array[Option[Vector[3]]] = Array.tabulate(vertexCount) { vertex =>
      project(IndexedSeq.tabulate(dimension) { axis =>
        vertices(vertex * dimension + axis) * size * scales(axis)
      })
    }
    def corners(face: Int): IndexedSeq[Vector[3]] =
      cornerIndices(face).flatMap(projected(_))
    val kept = (0 until faceCount).filter(corners(_).size == IndexedMeshN.CornersPerFace)
    if kept.isEmpty then
      TriangleMeshData(Array.emptyFloatArray, Array.emptyIntArray, FaceTriangles.VertexStride)
    else
      require(
        kept.size.toLong * IndexedMeshN.FloatsPerFace <= Int.MaxValue,
        s"too many faces for one triangle mesh: ${kept.size}"
      )
      val triangles = Array.ofDim[Float](kept.size * IndexedMeshN.FloatsPerFace)
      val indices = Array.ofDim[Int](kept.size * IndexedMeshN.IndicesPerFace)
      kept.zipWithIndex.foreach { (face, slot) =>
        val mesh = FaceTriangles.faceToTriangleMesh(corners(face))
        System.arraycopy(mesh.vertices, 0, triangles, slot * IndexedMeshN.FloatsPerFace,
          IndexedMeshN.FloatsPerFace)
        mesh.indices.zipWithIndex.foreach { (index, i) =>
          indices(slot * IndexedMeshN.IndicesPerFace + i) =
            index + slot * IndexedMeshN.CornersPerFace
        }
      }
      FaceTriangles.translateMesh(
        TriangleMeshData(triangles, indices, FaceTriangles.VertexStride), center
      )

  private def cornerIndices(face: Int): IndexedSeq[Int] =
    IndexedSeq.tabulate(IndexedMeshN.CornersPerFace)(corner =>
      quads(face * IndexedMeshN.CornersPerFace + corner)
    )

  private def coordinateSum(face: Int, axis: Int): Float =
    cornerIndices(face).map(vertex => vertices(vertex * dimension + axis)).sum

object IndexedMeshN:
  val CornersPerFace: Int = 4

  // a square is a fan of two triangles over its four corners
  private val TrianglesPerFace = 2
  private val IndicesPerTriangle = 3
  private val FloatsPerFace = CornersPerFace * FaceTriangles.VertexStride
  private val IndicesPerFace = TrianglesPerFace * IndicesPerTriangle

  // the centre square of a face split into 3 x 3 is a third of its edge
  private val CapDivisor = 3f
