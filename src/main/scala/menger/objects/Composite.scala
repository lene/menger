package menger.objects

import scala.util.Try

import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import menger.ProfilingConfig
import menger.RotationProjectionParameters
import menger.common.Patterns
import menger.input.Observer

class Composite(
  center: Vector3 = Vector3.Zero,
  scale: Float = 1f,
  geometries: List[Geometry]
)(using val profilingConfig: ProfilingConfig) extends Geometry(center, scale) with Observer:

  override def getModel: List[ModelInstance] =
    logTime("getModel()") {
      geometries.flatMap(_.getModel)
    }

  override def handleEvent(event: RotationProjectionParameters): Unit =
    geometries.foreach {
      case obs: Observer => obs.handleEvent(event)
      case _ => // Non-observer geometries don't need events
    }

  override def toString: String =
    s"Composite(${geometries.map(_.toString).mkString(", ")})"

object Composite:

  def parseCompositeFromCLIOption(
    spongeType: String,
    level: Float,
    material: Material,
    primitiveType: Int,
    rotationProjection: RotationProjectionParameters,
    createGeometry: (String, Float, Material, Int, RotationProjectionParameters) => Try[Geometry]
  )(using ProfilingConfig): Try[Geometry] =
    spongeType match
      case Patterns.CompositeType(content) =>
        val componentTypes = Patterns.parseCompositeComponents(content)
        val geometries = componentTypes.map(componentType =>
          createGeometry(componentType, level, material, primitiveType, rotationProjection)
        )

        // Check if all components were created successfully
        val failures = geometries.collect { case scala.util.Failure(ex) => ex }
        if failures.nonEmpty then
          scala.util.Failure(failures.head)
        else
          val successfulGeometries = geometries.collect { case scala.util.Success(geom) => geom }
          Try(Composite(Vector3.Zero, 1f, successfulGeometries))
      case _ => scala.util.Failure(IllegalArgumentException(s"Not a composite type: $spongeType"))