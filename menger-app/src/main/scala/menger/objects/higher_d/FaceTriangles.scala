package menger.objects.higher_d

import menger.common.TriangleMeshData
import menger.common.Vector
import menger.common.x
import menger.common.y
import menger.common.z

/** How one projected face becomes triangles, shared by every mesh that draws faces through a
  * projection ([[ProjectedMesh]], [[IndexedMeshN]]) so that they emit the same triangles. */
private[higher_d] object FaceTriangles:
  /** Floats per vertex: position(3), normal(3), uv(2). */
  val VertexStride: Int = 8


  /** A fan of `face.size - 2` triangles over the corners, which carry the face's Newell normal
    * and the UVs of a square. */
  def faceToTriangleMesh(face: IndexedSeq[Vector[3]]): TriangleMeshData =
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

    val verts = Array.tabulate(vpf * VertexStride) { idx =>
      val j = idx / VertexStride
      val v = face(j)
      idx % VertexStride match
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

    TriangleMeshData(verts, indices, VertexStride)

  def translateMesh(mesh: TriangleMeshData, offset: Vector[3]): TriangleMeshData =
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
