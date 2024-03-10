package menger.objects.higher_d

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.{Material, Model, ModelInstance}
import com.badlogic.gdx.math.Vector3
import menger.objects.{Builder, Geometry}

case class RotatedProjection(
  tesseract: Tesseract, projection: Projection, rotation: Rotation = Rotation(),
  material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(material, primitiveType) with RectMesh:

  lazy val projectedFaceVertices: Seq[RectVertices3D] = 
    tesseract.faces.map {rotation(_)}.map {projection(_)}
    
  lazy val projectedFaceInfo: Seq[RectInfo] = projectedFaceVertices.map {
    case (a, b, c, d) => (VertexInfo(a), VertexInfo(b), VertexInfo(c), VertexInfo(d))
  }

  lazy val mesh: Model = logTime("mesh") { model(projectedFaceInfo, primitiveType, material) }

  override def at(center: Vector3, scale: Float): List[ModelInstance] = ModelInstance(mesh) :: Nil
