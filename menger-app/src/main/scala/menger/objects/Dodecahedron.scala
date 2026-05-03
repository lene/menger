package menger.objects

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import menger.common.TriangleMeshData
import menger.common.TriangleMeshSource

case class Dodecahedron(
  center: Vector3 = Vector3.Zero,
  scale: Float = 1f,
  material: Material = Builder.WHITE_MATERIAL,
  primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(center, scale) with TriangleMeshSource:

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  def getModel: List[ModelInstance] =
    throw UnsupportedOperationException("Dodecahedron is OptiX-only; getModel not supported")

  def toTriangleMesh: TriangleMeshData = buildMesh

  private def buildMesh: TriangleMeshData =
    // Icosahedron vertices (unit sphere) — same as Icosahedron.scala
    val icoVerts: Array[(Float, Float, Float)] = Array(
      (0f,           0.5257311f,   0.8506508f),  // 0
      (0f,          -0.5257311f,   0.8506508f),  // 1
      (0f,           0.5257311f,  -0.8506508f),  // 2
      (0f,          -0.5257311f,  -0.8506508f),  // 3
      (0.5257311f,   0.8506508f,   0f),           // 4
      (-0.5257311f,  0.8506508f,   0f),           // 5
      (0.5257311f,  -0.8506508f,   0f),           // 6
      (-0.5257311f, -0.8506508f,   0f),           // 7
      (0.8506508f,   0f,           0.5257311f),   // 8
      (-0.8506508f,  0f,           0.5257311f),   // 9
      (0.8506508f,   0f,          -0.5257311f),   // 10
      (-0.8506508f,  0f,          -0.5257311f)    // 11
    )
    // 20 triangular faces — exactly as in Icosahedron.scala (CCW from outside)
    val icoFaces: Array[(Int, Int, Int)] = Array(
      (0, 8, 4),
      (0, 1, 8),
      (0, 4, 5),
      (0, 5, 9),
      (0, 9, 1),
      (4, 2, 5),
      (5, 11, 9),
      (9, 7, 1),
      (1, 6, 8),
      (8, 10, 4),
      (2, 11, 5),
      (11, 7, 9),
      (7, 6, 1),
      (6, 10, 8),
      (10, 2, 4),
      (3, 11, 2),
      (3, 7, 11),
      (3, 6, 7),
      (3, 10, 6),
      (3, 2, 10)
    )

    // Dodecahedron vertex i = centroid of icosahedron face i, projected to unit sphere
    val dodVerts: Array[(Float, Float, Float)] = icoFaces.map { (a, b, c) =>
      val (ax, ay, az) = icoVerts(a)
      val (bx, by, bz) = icoVerts(b)
      val (cx, cy, cz) = icoVerts(c)
      val mx = (ax + bx + cx) / 3f
      val my = (ay + by + cy) / 3f
      val mz = (az + bz + cz) / 3f
      val len = math.sqrt(mx * mx + my * my + mz * mz).toFloat
      (mx / len, my / len, mz / len)
    }

    // For each icosahedron vertex, collect the dodecahedron vertex indices (face centroids)
    // that contain it — these form a pentagonal face of the dodecahedron
    val icoVertToDodVerts: Array[List[Int]] =
      (0 until 12).toArray.map { icoV =>
        icoFaces.zipWithIndex.collect {
          case ((a, b, c), fi) if a == icoV || b == icoV || c == icoV => fi
        }.toList
      }

    // Sort pentagon vertices CCW when viewed from outside (along the icosahedron vertex direction)
    def sortPentagon(icoVertIdx: Int, faceIndices: List[Int]): Array[Int] =
      val (pvx, pvy, pvz) = icoVerts(icoVertIdx)
      // Arbitrary "up" vector perpendicular to pv
      val (upx, upy, upz) =
        if math.abs(pvx) < 0.9f then (1f, 0f, 0f) else (0f, 1f, 0f)
      // right = pv × up (normalized)
      val (rx, ry, rz) =
        val x = pvy * upz - pvz * upy
        val y = pvz * upx - pvx * upz
        val z = pvx * upy - pvy * upx
        val len = math.sqrt(x * x + y * y + z * z).toFloat
        (x / len, y / len, z / len)
      // forward = pv × right
      val (fx, fy, fz) =
        val x = pvy * rz - pvz * ry
        val y = pvz * rx - pvx * rz
        val z = pvx * ry - pvy * rx
        (x, y, z)
      faceIndices.sortBy { fi =>
        val (vx, vy, vz) = dodVerts(fi)
        val projR = vx * rx + vy * ry + vz * rz
        val projF = vx * fx + vy * fy + vz * fz
        math.atan2(projF, projR)
      }.toArray

    val pentagons: Array[Array[Int]] = icoVertToDodVerts.zipWithIndex.map {
      (faceList, icoIdx) => sortPentagon(icoIdx, faceList)
    }

    val cx = center.x
    val cy = center.y
    val cz = center.z

    // Fan-triangulate pentagons: (p0,p1,p2), (p0,p2,p3), (p0,p3,p4)
    val faces: Array[(Int, Int, Int)] = pentagons.flatMap { p =>
      Array((p(0), p(1), p(2)), (p(0), p(2), p(3)), (p(0), p(3), p(4)))
    }

    PolytopeUtil.flatShaded(faces, dodVerts, scale, cx, cy, cz)
