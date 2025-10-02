package menger.objects

import scala.util.Try

import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import menger.RotationProjectionParameters

class Composite(
  center: Vector3 = Vector3.Zero,
  scale: Float = 1f,
  geometries: List[Geometry]
) extends Geometry(center, scale):

  override def getModel: List[ModelInstance] =
    logTime("getModel()", 5) {
      geometries.flatMap(_.getModel)
    }

  override def handleEvent(event: RotationProjectionParameters): Unit =
    geometries.foreach(_.handleEvent(event))

  override def toString: String =
    s"Composite(${geometries.map(_.toString).mkString(", ")})"

object Composite:
  private val compositePattern = """composite\[(.+)]""".r

  def parseCompositeFromCLIOption(
    spongeType: String, level: Float, material: Material, primitiveType: Int,
    generateObject: (String, Float, Material, Int) => Try[Geometry]
  ): Try[Geometry] =
    spongeType match
      case compositePattern(content) =>
        val componentTypes = content.split(",").toList
        val geometries = componentTypes.map(componentType =>
          generateObject(componentType, level, material, primitiveType)
        )

        // Check if all components were created successfully
        val failures = geometries.collect { case scala.util.Failure(ex) => ex }
        if failures.nonEmpty then
          scala.util.Failure(failures.head)
        else
          val successfulGeometries = geometries.collect { case scala.util.Success(geom) => geom }
          Try(Composite(Vector3.Zero, 1f, successfulGeometries))
      case _ => scala.util.Failure(IllegalArgumentException(s"Not a composite type: $spongeType"))