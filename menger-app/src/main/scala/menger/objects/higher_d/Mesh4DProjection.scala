package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import menger.common.TriangleMeshData
import menger.common.TriangleMeshSource
import menger.common.Vector

/** Renders any 4D mesh by projecting it to 3D space.
  *
  * @param mesh4D The 4D mesh to project
  * @param center Center position of the projected mesh in 3D space
  * @param eyeW Distance from eye to projection hyperplane in 4D (must be > screenW)
  * @param screenW Distance from origin to projection hyperplane in 4D (must be > 0)
  * @param rotXW Rotation angle in XW plane (degrees)
  * @param rotYW Rotation angle in YW plane (degrees)
  * @param rotZW Rotation angle in ZW plane (degrees)
  */
case class Mesh4DProjection(
    mesh4D: Mesh4D,
    center: Vector3 = Vector3(0f, 0f, 0f),
    eyeW: Float = 3.0f,
    screenW: Float = 1.5f,
    rotXW: Float = 15f,
    rotYW: Float = 10f,
    rotZW: Float = 0f
) extends TriangleMeshSource:

  require(eyeW > screenW, s"eyeW ($eyeW) must be greater than screenW ($screenW)")
  require(eyeW > 0 && screenW > 0, "eyeW and screenW must be positive")

  private val rotation: Rotation =
    if rotXW == 0f && rotYW == 0f && rotZW == 0f then Rotation.identity
    else Rotation(rotXW, rotYW, rotZW, Vector[4](0f, 0f, 0f, 0f))

  private val projection = Projection(eyeW, screenW)

  private def projectedQuads: Seq[Quad3D] =
    mesh4D.faces.map { face4d =>
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
    // Compute mesh centroid from all projected quad vertices so each quad
    // can check whether its normal points outward or inward
    val allVertices = quads.flatMap(q => Seq(q(0), q(1), q(2), q(3)))
    val n = allVertices.length
    val meshCentroid = (
      allVertices.map(_.x).sum / n,
      allVertices.map(_.y).sum / n,
      allVertices.map(_.z).sum / n
    )
    val meshDataList = quads.map(quadToTriangleMesh(_, meshCentroid))
    val merged = TriangleMeshData.merge(meshDataList)
    translateMesh(merged, center)

  private def quadToTriangleMesh(
    quad: Quad3D,
    meshCentroid: (Float, Float, Float)
  ): TriangleMeshData =
    val v0 = quad(0)
    val v1 = quad(1)
    val v2 = quad(2)
    val v3 = quad(3)

    // Calculate face normal (cross product of two edges)
    val edge1 = new Vector3(v1).sub(v0)
    val edge2 = new Vector3(v3).sub(v0)
    val normalVec = new Vector3(edge1).crs(edge2).nor()

    // Handle degenerate faces (zero-area when projected edge-on)
    val isDegenerate = normalVec.len2() < 0.0001f
    val (rawNx, rawNy, rawNz) =
      if isDegenerate then (0f, 1f, 0f)
      else (normalVec.x, normalVec.y, normalVec.z)

    // Ensure the normal points outward from the mesh centroid.
    // Perspective projection of 4D faces does not guarantee consistent winding,
    // so we check and flip if the normal points toward the centroid instead.
    val (fnx, fny, fnz, pa, pb, pc, pd) =
      if isDegenerate then
        (rawNx, rawNy, rawNz, v0, v1, v2, v3)
      else
        val (mcx, mcy, mcz) = meshCentroid
        val quadCx = (v0.x + v1.x + v2.x + v3.x) / 4f
        val quadCy = (v0.y + v1.y + v2.y + v3.y) / 4f
        val quadCz = (v0.z + v1.z + v2.z + v3.z) / 4f
        val outwardDot =
          rawNx * (quadCx - mcx) +
          rawNy * (quadCy - mcy) +
          rawNz * (quadCz - mcz)
        if outwardDot >= 0f then
          (rawNx, rawNy, rawNz, v0, v1, v2, v3)
        else
          // Flip normal and reverse winding: swap v1 <-> v3
          (-rawNx, -rawNy, -rawNz, v0, v3, v2, v1)

    // Vertex format: position(3) + normal(3) + uv(2) = 8 floats
    val vertices = Array(
      pa.x, pa.y, pa.z, fnx, fny, fnz, 0f, 0f,
      pb.x, pb.y, pb.z, fnx, fny, fnz, 1f, 0f,
      pc.x, pc.y, pc.z, fnx, fny, fnz, 1f, 1f,
      pd.x, pd.y, pd.z, fnx, fny, fnz, 0f, 1f
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

/** Backward-compatible factory for creating tesseract meshes */
object TesseractMesh:
  def apply(
      center: Vector3 = Vector3(0f, 0f, 0f),
      size: Float = 1.0f,
      eyeW: Float = 3.0f,
      screenW: Float = 1.5f,
      rotXW: Float = 15f,
      rotYW: Float = 10f,
      rotZW: Float = 0f
  ): Mesh4DProjection =
    Mesh4DProjection(
      mesh4D = Tesseract(size = size),
      center = center,
      eyeW = eyeW,
      screenW = screenW,
      rotXW = rotXW,
      rotYW = rotYW,
      rotZW = rotZW
    )
