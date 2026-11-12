package menger.objects.higher_d

import menger.common.TriangleMeshData
import menger.common.TriangleMeshSource
import menger.common.Vector

/** A mesh of any dimension drawn as a 3D triangle mesh: each vertex goes through
  * `projectVertex`, the faces are triangulated and the result is moved to `center`. A face with
  * a clipped vertex (`isClipped`) is left out. */
trait ProjectedMesh[N <: Int & Singleton] extends TriangleMeshSource:
  def sourceMesh: Mesh[N]
  def center: Vector[3]

  protected def projectVertex(vertex: Vector[N]): Vector[3]

  /** True for a vertex that has no valid projection (at or behind a stage's eye). */
  protected def isClipped(vertex: Vector[N]): Boolean = false

  private def projectedVertices: Seq[IndexedSeq[Vector[3]]] =
    sourceMesh.faces.filterNot(face => face.vertices.exists(isClipped)).map { face =>
      val vpf = face.vertsPerFace
      (0 until vpf).map(i => projectVertex(face(i))).toIndexedSeq
    }

  override def toTriangleMesh: TriangleMeshData =
    if sourceMesh.vertsPerFace < 3 then
      TriangleMeshData(Array.emptyFloatArray, Array.emptyIntArray, FaceTriangles.VertexStride)
    else
      val faces = projectedVertices
      val meshDataList = faces.map(FaceTriangles.faceToTriangleMesh)
      val merged = TriangleMeshData.merge(meshDataList)
      FaceTriangles.translateMesh(merged, center)
