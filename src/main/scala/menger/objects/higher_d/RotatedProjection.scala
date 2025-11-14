package menger.objects.higher_d

import java.util.concurrent.atomic.AtomicReference

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.Model
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import menger.RotationProjectionParameters
import menger.input.Observer
import menger.objects.Builder
import menger.objects.Geometry

case class RotatedProjectionState(
  projection: Projection,
  rotation: Rotation,
  cachedMesh: Option[Model]
)

case class RotatedProjection(
  center: Vector3 = Vector3.Zero, scale: Float = 1f, object4D: Mesh4D,
  initialProjection: Projection,
  initialRotation: Rotation = Rotation(),
  material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
)(using val profilingConfig: menger.ProfilingConfig) extends Geometry(center, scale) with RectMesh with Observer:

  // Encapsulated mutable state using AtomicReference
  private val state = AtomicReference(
    RotatedProjectionState(initialProjection, initialRotation, None)
  )

  private def currentState: RotatedProjectionState = state.get()

  def projection: Projection = currentState.projection
  def rotation: Rotation = currentState.rotation

  def projectedFaceVertices: Seq[Quad3D] =
    val s = currentState
    object4D.faces.map {s.rotation(_)}.map {s.projection(_)}

  def projectedFaceInfo: Seq[QuadInfo] = projectedFaceVertices.map {
    rv => QuadInfo(VertexInfo(rv(0)), VertexInfo(rv(1)), VertexInfo(rv(2)), VertexInfo(rv(3)))
  }

  def mesh: Model = logTime("mesh") {
    val s = currentState
    s.cachedMesh match
      case Some(cached) => cached
      case None =>
        val newMesh = model(projectedFaceInfo, primitiveType, material)
        state.updateAndGet(_.copy(cachedMesh = Some(newMesh)))
        newMesh
  }

  override def getModel: List[ModelInstance] = ModelInstance(mesh) :: Nil

  override def handleEvent(event: RotationProjectionParameters): Unit =
    state.updateAndGet { s =>
      RotatedProjectionState(
        s.projection + event.projection,
        s.rotation + event.rotation,
        None  // Invalidate cache when state changes
      )
    }

  override def toString: String = s"${getClass.getSimpleName}[$object4D]"

object RotatedProjection:
  def apply(
    object4D: Mesh4D, parameters: RotationProjectionParameters,
    material: Material, primitiveType: Int
  )(using config: menger.ProfilingConfig): RotatedProjection =
    RotatedProjection(Vector3.Zero, 1f, object4D, parameters.projection, parameters.rotation, material, primitiveType)
