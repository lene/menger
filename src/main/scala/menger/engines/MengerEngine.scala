package menger.engines

import scala.util.Try

import com.badlogic.gdx.Game
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.ModelInstance
import menger.GDXResources
import menger.ProfilingConfig
import menger.RotationProjectionParameters
import menger.common.Const
import menger.objects.Builder
import menger.objects.Geometry
import menger.objects.GeometryFactory

abstract class MengerEngine(
  val spongeType: String, val spongeLevel: Float,
  val rotationProjectionParameters: RotationProjectionParameters, val lines: Boolean, val color: Color,
  val faceColor: Option[Color] = None, val lineColor: Option[Color] = None,
  val fpsLogIntervalMs: Int = Const.fpsLogIntervalMs
)(using val profilingConfig: ProfilingConfig) extends Game:
  protected val material = Builder.material(color)
  protected lazy val primitiveType: Int = if lines then GL20.GL_LINES else GL20.GL_TRIANGLES
  protected val isOverlayMode: Boolean = faceColor.isDefined && lineColor.isDefined
  protected def gdxResources: GDXResources
  protected def drawables: List[ModelInstance]
  override def resume(): Unit = {}
  override def pause(): Unit = {}
  override def dispose(): Unit = gdxResources.dispose()
  override def resize(width: Int, height: Int): Unit = gdxResources.resize()
  def currentRotProj: RotationProjectionParameters = rotationProjectionParameters

  protected def generateObjectWithOverlay(spongeType: String, level: Float): Try[Geometry] =
    GeometryFactory.createWithOverlay(
      spongeType, level, faceColor, lineColor, material, currentRotProj
    )
