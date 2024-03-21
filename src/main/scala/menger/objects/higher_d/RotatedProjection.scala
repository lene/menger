package menger.objects.higher_d

import com.badlogic.gdx.Gdx

import scala.compiletime.uninitialized
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.{Material, Model, ModelInstance}
import com.badlogic.gdx.math.Vector3
import menger.RotationProjectionParameters
import menger.input.Observer
import menger.objects.{Builder, Geometry}

case class RotatedProjection(
  object4D: Mesh4D, var projection: Projection, var rotation: Rotation = Rotation(),
  material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(material, primitiveType) with RectMesh with Observer:

  private var changed = true
  private var precomputedMesh: Model = uninitialized

  def projectedFaceVertices: Seq[RectVertices3D] =
    object4D.faces.map {rotation(_)}.map {projection(_)}
    
  def projectedFaceInfo: Seq[RectInfo] = projectedFaceVertices.map {
    case (a, b, c, d) => (VertexInfo(a), VertexInfo(b), VertexInfo(c), VertexInfo(d))
  }

  def mesh: Model = logTime("mesh", 10) {
    if changed then
      precomputedMesh = model(projectedFaceInfo, primitiveType, material)
      changed = false
    precomputedMesh
  }

  override def at(center: Vector3, scale: Float): List[ModelInstance] = ModelInstance(mesh) :: Nil

  override def handleEvent(event: RotationProjectionParameters): Unit =
    rotation += event.rotation
    projection += event.projection
    changed = true

object RotatedProjection:
  def apply(
    object4D: Mesh4D, parameters: RotationProjectionParameters,
    material: Material, primitiveType: Int
  ): RotatedProjection =
    RotatedProjection(object4D, parameters.projection, parameters.rotation, material, primitiveType)
