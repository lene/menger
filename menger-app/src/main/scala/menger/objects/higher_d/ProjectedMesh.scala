package menger.objects.higher_d

import menger.common.TriangleMeshData
import menger.common.TriangleMeshSource
import menger.common.Vector
import menger.common.x
import menger.common.y
import menger.common.z

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
      TriangleMeshData(Array.emptyFloatArray, Array.emptyIntArray, vertexStride = 8)
    else
      val faces = projectedVertices
      val meshDataList = faces.map(faceToTriangleMesh)
      val merged = TriangleMeshData.merge(meshDataList)
      translateMesh(merged, center)

  private def faceToTriangleMesh(face: IndexedSeq[Vector[3]]): TriangleMeshData =
    val vpf = face.size
    val triCount = vpf - 2

    // Newell's method for face normal
    val (nxAcc, nyAcc, nzAcc) = (0 until vpf).foldLeft((0.0, 0.0, 0.0)) {
      case ((nx, ny, nz), i) =>
        val cur = face(i)
        val nxt = face((i + 1) % vpf)
        (nx + (cur.y - nxt.y).toDouble * (cur.z + nxt.z).toDouble,
         ny + (cur.z - nxt.z).toDouble * (cur.x + nxt.x).toDouble,
         nz + (cur.x - nxt.x).toDouble * (cur.y + nxt.y).toDouble)
    }
    val nl = math.sqrt(nxAcc * nxAcc + nyAcc * nyAcc + nzAcc * nzAcc).toFloat
    // Newell's-method magnitude scales with face area, i.e. edge^2, so nl
    // scales with edge^2. A fixed absolute epsilon here misclassifies
    // small-but-valid faces as degenerate (e.g. deep fractal recursion,
    // where faces shrink by 3x per level) -- scale the threshold by the
    // face's own edge length instead. (Mirrors the fix in optix-jni's
    // project4d.cu, which computes the analogous normal on the GPU.)
    val edgeLen = (face(1) - face(0)).len
    val (ennX, ennY, ennZ) =
      if edgeLen < 1e-6f || nl < 1e-3f * edgeLen * edgeLen then (0f, 1f, 0f)
      else ((nxAcc / nl).toFloat, (nyAcc / nl).toFloat, (nzAcc / nl).toFloat)

    val verts = Array.tabulate(vpf * 8) { idx =>
      val j = idx / 8
      val v = face(j)
      idx % 8 match
        case 0 => v.x
        case 1 => v.y
        case 2 => v.z
        case 3 => ennX
        case 4 => ennY
        case 5 => ennZ
        case 6 => if j == 0 || j == 3 then 0f else 1f
        case _ => if j < 2 then 0f else 1f
    }

    val indices = Array.tabulate(triCount * 3) { i =>
      val k = i / 3 + 2
      i % 3 match
        case 0 => 0
        case 1 => k - 1
        case _ => k
    }

    TriangleMeshData(verts, indices, vertexStride = 8)

  private def translateMesh(mesh: TriangleMeshData, offset: Vector[3]): TriangleMeshData =
    if offset.x == 0f && offset.y == 0f && offset.z == 0f then mesh
    else
      val translated = mesh.vertices.zipWithIndex.map { case (value, index) =>
        val positionInVertex = index % mesh.vertexStride
        positionInVertex match
          case 0 => value + offset.x
          case 1 => value + offset.y
          case 2 => value + offset.z
          case _ => value
      }
      TriangleMeshData(translated, mesh.indices, mesh.vertexStride)
