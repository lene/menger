package menger

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.utils.Timer
import com.typesafe.scalalogging.LazyLogging
import menger.input.EventDispatcher
import menger.objects.Geometry


class InteractiveMengerEngine(
  spongeType: String = "square", spongeLevel: Float = 0.0f,
  rotationProjectionParameters: RotationProjectionParameters = RotationProjectionParameters(),
  lines: Boolean = false, color: Color = Color.WHITE, timeout: Float = 0,
  faceColor: Option[Color] = None, lineColor: Option[Color] = None,
  fpsLogIntervalMs: Int = 1000
)(using config: ProfilingConfig) extends MengerEngine(spongeType, spongeLevel, rotationProjectionParameters, lines, color, faceColor, lineColor, fpsLogIntervalMs)
with LazyLogging:
  private lazy val sponge: Geometry = generateObjectWithOverlay(
    spongeType, spongeLevel
  ) match
    case scala.util.Success(geometry) => geometry
    case scala.util.Failure(exception) =>
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

  private def startExitTimer(timeout: Float): Unit =
    logger.info(s"Starting timer for $timeout seconds")
    Timer.schedule(() => Gdx.app.exit(), timeout, 0)

  private def dispatcherWithRegisteredSponge =
    val dispatcher = EventDispatcher()
    "tesseract".r.findFirstIn(spongeType).foreach(_ => dispatcher.addObserver(sponge))
    dispatcher
