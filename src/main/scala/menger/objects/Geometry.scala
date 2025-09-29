package menger.objects

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import menger.RotationProjectionParameters
import menger.input.Observer

trait Geometry extends Observer:
  def at(center: Vector3, scale: Float = 1): List[ModelInstance]
  override def toString: String = getClass.getSimpleName
  override def handleEvent(event: RotationProjectionParameters): Unit = {}
  
  def logTime[T](msg: String, minDuration: Int = 0)(f: => T): T =
    val start = System.nanoTime()
    val result = f
    val duration = (System.nanoTime() - start) / 1_000_000
    if duration >= minDuration && Gdx.app != null then
    Gdx.app.log(s"${getClass.getSimpleName}.$msg", s"${duration}ms")
    result

  def composite(other: Geometry): Geometry =
    ???
    