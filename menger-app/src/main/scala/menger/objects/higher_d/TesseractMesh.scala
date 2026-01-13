package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import menger.common.TriangleMeshData
import menger.common.TriangleMeshSource
import menger.common.Vector

case class TesseractMesh(
    center: Vector3 = Vector3(0f, 0f, 0f),
    size: Float = 1.0f,
    eyeW: Float = 3.0f,
    screenW: Float = 1.5f,
    rotXW: Float = 15f,
    rotYW: Float = 10f,
    rotZW: Float = 0f
) extends TriangleMeshSource:

  require(eyeW > screenW, s"eyeW ($eyeW) must be greater than screenW ($screenW)")
  require(eyeW > 0 && screenW > 0, "eyeW and screenW must be positive")

  private val tesseract = Tesseract(size = size)

  private val rotation: Rotation =
    if rotXW == 0f && rotYW == 0f && rotZW == 0f then Rotation.identity
    else Rotation(rotXW, rotYW, rotZW, Vector[4](0f, 0f, 0f, 0f))

  private val projection = Projection(eyeW, screenW)

  private def projectedQuads: Seq[Quad3D] =
    tesseract.faces.map { face4d =>
      val rotatedFace = Face4D(
        rotation(face4d.a),
        rotation(face4d.b),
        rotation(face4d.c),
        rotation(face4d.d)
      )
      projection(rotatedFace)
    }

  override def toTriangleMesh: TriangleMeshData =
    val quads = projectedQuads
    val meshDataList = quads.map(quadToTriangleMesh)
    val merged = TriangleMeshData.merge(meshDataList)
    translateMesh(merged, center)

  private def quadToTriangleMesh(quad: Quad3D): TriangleMeshData =
    val v0 = quad(0)
    val v1 = quad(1)
    val v2 = quad(2)
    val v3 = quad(3)

    // Calculate face normal (cross product of two edges)
    val edge1 = new Vector3(v1).sub(v0)
    val edge2 = new Vector3(v3).sub(v0)
    val normal = new Vector3(edge1).crs(edge2).nor()

    // Handle degenerate faces (zero-area when projected edge-on)
    val (nx, ny, nz) =
      if normal.len2() < 0.0001f then (0f, 1f, 0f)
      else (normal.x, normal.y, normal.z)

    // Vertex format: position(3) + normal(3) + uv(2) = 8 floats
    val vertices = Array(
      v0.x, v0.y, v0.z, nx, ny, nz, 0f, 0f,
      v1.x, v1.y, v1.z, nx, ny, nz, 1f, 0f,
      v2.x, v2.y, v2.z, nx, ny, nz, 1f, 1f,
      v3.x, v3.y, v3.z, nx, ny, nz, 0f, 1f
    )

    // Two triangles: (v0,v1,v2) and (v0,v2,v3)
    val indices = Array(0, 1, 2, 0, 2, 3)

    TriangleMeshData(vertices, indices, vertexStride = 8)

  private def translateMesh(mesh: TriangleMeshData, offset: Vector3): TriangleMeshData =
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
