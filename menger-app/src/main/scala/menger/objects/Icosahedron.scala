package menger.objects

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import menger.common.TriangleMeshData
import menger.common.TriangleMeshSource

case class Icosahedron(
  center: Vector3 = Vector3.Zero,
  scale: Float = 1f,
  material: Material = Builder.WHITE_MATERIAL,
  primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(center, scale) with TriangleMeshSource:

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  def getModel: List[ModelInstance] =
    throw UnsupportedOperationException("Icosahedron is OptiX-only; getModel not supported")

  def toTriangleMesh: TriangleMeshData =
    // Unit icosahedron vertices: three mutually perpendicular golden rectangles,
    // normalized by √(1 + φ²)
    val verts: Array[(Float, Float, Float)] = Array(
      (0f,          0.5257311f,  0.8506508f),   // 0
      (0f,         -0.5257311f,  0.8506508f),   // 1
      (0f,          0.5257311f, -0.8506508f),   // 2
      (0f,         -0.5257311f, -0.8506508f),   // 3
      ( 0.5257311f, 0.8506508f,  0f),            // 4
      (-0.5257311f, 0.8506508f,  0f),            // 5
      ( 0.5257311f,-0.8506508f,  0f),            // 6
      (-0.5257311f,-0.8506508f,  0f),            // 7
      ( 0.8506508f, 0f,          0.5257311f),    // 8
      (-0.8506508f, 0f,          0.5257311f),    // 9
      ( 0.8506508f, 0f,         -0.5257311f),    // 10
      (-0.8506508f, 0f,         -0.5257311f)     // 11
    )
    // 20 triangular faces wound CCW from outside
    val faces: Array[(Int, Int, Int)] = Array(
      (0,8,4),  (0,1,8),  (0,4,5),  (0,5,9),  (0,9,1),   // top ring
      (4,2,5),  (5,11,9), (9,7,1),  (1,6,8),  (8,10,4),  // upper equatorial
      (2,11,5), (11,7,9), (7,6,1),  (6,10,8), (10,2,4),  // lower equatorial
      (3,11,2), (3,7,11), (3,6,7),  (3,10,6), (3,2,10)   // bottom ring
    )
    PolytopeUtil.flatShaded(faces, verts, scale, center.x, center.y, center.z)
