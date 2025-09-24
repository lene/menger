package menger

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3
import com.badlogic.gdx.utils.Timer
import com.typesafe.scalalogging.LazyLogging
import menger.input.EventDispatcher
import menger.objects.Geometry


class InteractiveMengerEngine(
  spongeType: String = "square", spongeLevel: Int = 0,
  rotationProjectionParameters: RotationProjectionParameters = RotationProjectionParameters(),
  lines: Boolean = false, color: Color = Color.WHITE, timeout: Float = 0
) extends MengerEngine(spongeType, spongeLevel, rotationProjectionParameters, lines, color)
with LazyLogging:
  private lazy val sponge: Geometry = generateObject(spongeType, spongeLevel, material, primitiveType)
  private lazy val eventDispatcher = EventDispatcher()
  "tesseract".r.findFirstIn(spongeType).foreach(_ => eventDispatcher.addObserver(sponge))

  protected def drawables: List[ModelInstance] = sponge.at(Vector3(0, 0, 0), 1)
  protected def gdxResources: GDXResources = GDXResources(Some(eventDispatcher))

  override def create(): Unit =
    Gdx.app.log(s"${getClass.getSimpleName}.create()", s"$sponge color=$color")
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

