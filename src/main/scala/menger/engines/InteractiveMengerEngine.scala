package menger.engines

import scala.util.Failure
import scala.util.Success

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.utils.Timer
import com.typesafe.scalalogging.LazyLogging
import menger.GDXResources
import menger.ProfilingConfig
import menger.RotationProjectionParameters
import menger.common.Const
import menger.input.EventDispatcher
import menger.objects.Geometry


class InteractiveMengerEngine(
  spongeType: String = "square", spongeLevel: Float = 0.0f,
  rotationProjectionParameters: RotationProjectionParameters = RotationProjectionParameters(),
  lines: Boolean = false, color: Color = Color.WHITE, val timeout: Float = 0,
  faceColor: Option[Color] = None, lineColor: Option[Color] = None,
  fpsLogIntervalMs: Int = Const.fpsLogIntervalMs
)(using config: ProfilingConfig) extends MengerEngine(spongeType, spongeLevel, rotationProjectionParameters, lines, color, faceColor, lineColor, fpsLogIntervalMs)
with TimeoutSupport with LazyLogging:
  private lazy val sponge: Geometry = generateObjectWithOverlay(
    spongeType, spongeLevel
  ) match
    case Success(geometry) => geometry
    case Failure(exception) =>
      logger.error(s"Failed to create sponge type '$spongeType': ${exception.getMessage}")
      sys.exit(1)
  private lazy val eventDispatcher = dispatcherWithRegisteredSponge

  protected def drawables: List[ModelInstance] =
    given ProfilingConfig = profilingConfig
    sponge.getModel
  protected lazy val gdxResources: GDXResources = GDXResources(Some(eventDispatcher), fpsLogIntervalMs)

  override def create(): Unit =
    logger.info(s"$sponge color=$color")
    if timeout > 0 then startExitTimer(timeout)

  override def render(): Unit = gdxResources.render(drawables)

  override def pause(): Unit =
    super.pause()
    Timer.instance().stop()

  override def resume(): Unit =
    super.resume()
    Timer.instance().start()

  override def dispose(): Unit =
    super.dispose()
    Timer.instance().stop()

  private def dispatcherWithRegisteredSponge =
    val dispatcher = EventDispatcher()
    "tesseract".r.findFirstIn(spongeType).foreach(_ => dispatcher.addObserver(sponge))
    dispatcher
