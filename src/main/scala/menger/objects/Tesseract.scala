package menger.objects

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.VertexAttributes.Usage
import com.badlogic.gdx.graphics.g3d.utils.{MeshBuilder, MeshPartBuilder}
import com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo
import com.badlogic.gdx.graphics.g3d.{Material, Model, ModelInstance}
import com.badlogic.gdx.math.{Vector3, Vector4}

/** A tesseract of edge length `size` centered at the origin */
case class Tesseract(
  size: Float = 1.0,
  material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
):

  type RectIndices = (Int, Int, Int, Int)
  type RectVertices = (Vector4, Vector4, Vector4, Vector4)

  lazy val vertices: Seq[Vector4] = for (
    xx <- -1 to 1 by 2; yy <- -1 to 1 by 2; zz <- -1 to 1 by 2; ww <- -1 to 1 by 2
  ) yield Vector4(xx * size / 2, yy * size / 2, zz * size / 2, ww * size / 2)

  // order of vertices taken from
  // https://github.com/lene/HyperspaceExplorer/blob/master/src/Displayable/Object/ObjectImplementations.cpp#L55
  lazy val faceIndices: Seq[RectIndices] = Seq(
    ( 0, 1, 3, 2), ( 0, 1, 5, 4), ( 0, 1, 9, 8), ( 0, 2, 6, 4),
    ( 0, 2,10, 8), ( 0, 4,12, 8), ( 1, 3, 7, 5), ( 1, 3,11, 9),
    ( 1, 5,13, 9), ( 2, 3, 7, 6), ( 2, 3,11,10), ( 2, 6,14,10),
    ( 3, 7,15,11), ( 4, 5, 7, 6), ( 4, 5,13,12), ( 4, 6,14,12),
    ( 5, 7,15,13), ( 6, 7,15,14), ( 8, 9,11,10), ( 8, 9,13,12),
    ( 8,10,14,12), ( 9,11,15,13), (10,11,15,14), (12,13,15,14)
  )

  lazy val faces: Seq[RectVertices] = faceIndices.map {
    case (a, b, c, d) => (vertices(a), vertices(b), vertices(c), vertices(d))
  }

  lazy val edges: Seq[(Vector4, Vector4)] = faces.flatMap {
    // converting tuples to sets to make reversed edges equal to the original ones
    case (a, b, c, d) => Seq(Set(a, b), Set(b, c), Set(c, d), Set(d, a))
  }.distinct.map(set => (set.head, set.last))



/** project 4D points to 3D where the point we look at is at the origin `(0, 0, 0, 0)`,
 *  the eye is at `(0, 0, 0, -eyeW)` and the screen is at `(0, 0, 0, -screenW)`
 */
case class Projection(eyeW: Float, screenW: Float):
  assert(eyeW > 0 && screenW > 0, "eyeW and screenW must be positive")
  assert(eyeW > screenW, "eyeW must be greater than screenW")

  /** project a single 4D point `point` to 3D */
  def apply(point: Vector4): Vector3 =
    val projectionFactor = (eyeW - screenW) / (eyeW - point.w)
    Vector3(point.x * projectionFactor, point.y * projectionFactor, point.z * projectionFactor)

  /** project a sequence of 4D points to 3D */
  def apply(points: Seq[Vector4]): Seq[Vector3] = points.map(apply)
  def apply(points: (Vector4, Vector4, Vector4, Vector4)): (Vector3, Vector3, Vector3, Vector3) = (
    apply(points(0)), apply(points(1)), apply(points(2)), apply(points(3))
  )

case class TesseractProjection(
  tesseract: Tesseract, projection: Projection,
  material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(material, primitiveType):
  type RectVertices = (Vector3, Vector3, Vector3, Vector3)
  type RectInfo = (VertexInfo, VertexInfo, VertexInfo, VertexInfo)

  lazy val projectedFaceVertices: Seq[RectVertices] = tesseract.faces.map {projection(_)}
  lazy val projectedFaceInfo: Seq[RectInfo] = projectedFaceVertices.map {
    case (a, b, c, d) => (VertexInfo(a), VertexInfo(b), VertexInfo(c), VertexInfo(d))
  }

  lazy val mesh: Model = logTime("mesh") {
    Builder.modelBuilder.begin()
    projectedFaceInfo.grouped(MeshBuilder.MAX_VERTICES / 4).foreach(facesPart =>
      val meshBuilder: MeshPartBuilder = Builder.modelBuilder.part(
        "sponge", primitiveType, Usage.Position | Usage.Normal, material
      )
      facesPart.foreach(face => meshBuilder.rect.tupled(face))
    )
    Builder.modelBuilder.end()
  }

  override def at(center: Vector3, scale: Float): List[ModelInstance] = ModelInstance(mesh) :: Nil

object VertexInfo:
  def apply(v: Vector3): com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo =
    com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo().setPos(v)