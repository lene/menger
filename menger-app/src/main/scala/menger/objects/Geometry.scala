package menger.objects

import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.common.ProfilingConfig

trait Geometry(center: Vector3 = Vector3.Zero, scale: Float = 1f) extends LazyLogging:
  override def toString: String = getClass.getSimpleName

  inline def logTime[T](msg: String)(f: => T)(using config: ProfilingConfig): T =
    if config.isEnabled then
      val start = System.nanoTime()
      val result = f
      val duration = (System.nanoTime() - start) / 1_000_000
      if duration >= config.threshold then
        logger.debug(s"${getClass.getSimpleName}.$msg: ${duration}ms")
      result
    else
      f

