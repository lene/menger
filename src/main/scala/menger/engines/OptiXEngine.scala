package menger.engines

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.Pixmap
import com.badlogic.gdx.graphics.Texture
import com.badlogic.gdx.graphics.g2d.SpriteBatch
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.typesafe.scalalogging.LazyLogging
import menger.GDXResources
import menger.OptiXResources
import menger.ProfilingConfig
import menger.RotationProjectionParameters

@SuppressWarnings(Array("org.wartremover.warts.Throw", "org.wartremover.warts.Var"))
class OptiXEngine(
  spongeType: String,
  spongeLevel: Float,
  rotationProjectionParameters: RotationProjectionParameters,
  lines: Boolean,
  color: Color,
  faceColor: Option[Color],
  lineColor: Option[Color],
  fpsLogIntervalMs: Int,
  val sphereRadius: Float,
  val timeout: Float = 0f
)(using profilingConfig: ProfilingConfig) extends MengerEngine(
  spongeType, spongeLevel, rotationProjectionParameters, lines, color,
  faceColor, lineColor, fpsLogIntervalMs
) with TimeoutSupport with LazyLogging:

  private case class RenderState(
    texture: Option[Texture],
    pixmap: Option[Pixmap],
    width: Int,
    height: Int
  ):
    def dispose(): Unit =
      texture.foreach(_.dispose())
      pixmap.foreach(_.dispose())

  private lazy val optiXResources: OptiXResources = new OptiXResources(sphereRadius)
  private lazy val batch: SpriteBatch = new SpriteBatch()
  private var renderState: RenderState = RenderState(None, None, 0, 0)

  protected def drawables: List[ModelInstance] =
    throw new UnsupportedOperationException("OptiXEngine doesn't use drawables")

  protected def gdxResources: GDXResources =
    throw new UnsupportedOperationException("OptiXEngine doesn't use gdxResources")

  override def create(): Unit =
    logger.info(s"Creating OptiXEngine with sphere radius=$sphereRadius, color=$color")
    optiXResources.initialize()
    if timeout > 0 then startExitTimer(timeout)

  override def render(): Unit =
    Gdx.gl.glClearColor(0.25f, 0.25f, 0.25f, 1.0f)
    Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)

    val width = Gdx.graphics.getWidth
    val height = Gdx.graphics.getHeight
    val rgbaBytes = optiXResources.renderScene(width, height)
    renderToScreen(rgbaBytes, width, height)

  private def renderToScreen(rgbaBytes: Array[Byte], width: Int, height: Int): Unit =
    val needsRecreate = width != renderState.width || height != renderState.height

    if needsRecreate then
      renderState.dispose()
      renderState = RenderState(None, None, width, height)

    val pm = renderState.pixmap.getOrElse {
      val newPixmap = new Pixmap(width, height, Pixmap.Format.RGBA8888)
      renderState = renderState.copy(pixmap = Some(newPixmap))
      newPixmap
    }

    pm.getPixels.clear()
    pm.getPixels.put(rgbaBytes)
    pm.getPixels.rewind()

    val tex = renderState.texture match
      case Some(existingTexture) =>
        existingTexture.draw(pm, 0, 0)
        existingTexture
      case None =>
        val newTexture = new Texture(pm)
        renderState = renderState.copy(texture = Some(newTexture))
        newTexture

    batch.begin()
    batch.draw(tex, 0, 0, Gdx.graphics.getWidth.toFloat, Gdx.graphics.getHeight.toFloat)
    batch.end()

  override def resize(width: Int, height: Int): Unit =
    logger.debug(s"Window resized to ${width}x${height}")

  override def dispose(): Unit =
    logger.info("Disposing OptiXEngine")
    renderState.dispose()
    batch.dispose()
    optiXResources.dispose()
