package menger.objects

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import menger.ProfilingConfig
import menger.RotationProjectionParameters
import menger.input.Observer

trait Geometry(center: Vector3 = Vector3.Zero, scale: Float = 1f) extends Observer:
  def getModel: List[ModelInstance]
  override def toString: String = getClass.getSimpleName
  override def handleEvent(event: RotationProjectionParameters): Unit = {}

  inline def logTime[T](msg: String)(f: => T)(using config: ProfilingConfig): T =
    if config.isEnabled then
      val start = System.nanoTime()
      val result = f
      val duration = (System.nanoTime() - start) / 1_000_000
      if duration >= config.threshold then
        Option(Gdx.app).foreach(_.log(s"${getClass.getSimpleName}.$msg", s"${duration}ms"))
      result
    else
      f

