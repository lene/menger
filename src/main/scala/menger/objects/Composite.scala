package menger.objects

import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import menger.RotationProjectionParameters

class Composite(
  center: Vector3 = Vector3.Zero,
  scale: Float = 1f,
  geometries: List[Geometry]
) extends Geometry(center, scale):

  override def at(): List[ModelInstance] =
    logTime("at()") {
      geometries.flatMap(_.at())
    }

  override def handleEvent(event: RotationProjectionParameters): Unit =
    geometries.foreach(_.handleEvent(event))

  override def toString: String =
    s"Composite(${geometries.map(_.toString).mkString(", ")})"