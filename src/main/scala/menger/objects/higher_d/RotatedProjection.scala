package menger.objects.higher_d

import scala.compiletime.uninitialized

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.Model
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import menger.RotationProjectionParameters
import menger.input.Observer
import menger.objects.Builder
import menger.objects.Geometry

case class RotatedProjection(
  object4D: Mesh4D, var projection: Projection, var rotation: Rotation = Rotation(),
  material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(material, primitiveType) with RectMesh with Observer:

  private var changed = true
  private var precomputedMesh: Model = uninitialized

  def projectedFaceVertices: Seq[Quad3D] =
    object4D.faces.map {rotation(_)}.map {projection(_)}
    
  def projectedFaceInfo: Seq[QuadInfo] = projectedFaceVertices.map {
    rv => QuadInfo(VertexInfo(rv(0)), VertexInfo(rv(1)), VertexInfo(rv(2)), VertexInfo(rv(3)))
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

  override def toString: String = s"${getClass.getSimpleName}[$object4D]"

object RotatedProjection:
  def apply(
    object4D: Mesh4D, parameters: RotationProjectionParameters,
    material: Material, primitiveType: Int
  ): RotatedProjection =
    RotatedProjection(object4D, parameters.projection, parameters.rotation, material, primitiveType)
