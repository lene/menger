package menger

import java.util.concurrent.atomic.AtomicReference

import com.badlogic.gdx.graphics.g2d.SpriteBatch

case class OptiXRenderState(
  renderState: RenderState,
  hasSaved: Boolean,
  needsRender: Boolean
)

case class OptiXRenderResources(initialWidth: Int, initialHeight: Int):

  private val state = AtomicReference(
    OptiXRenderState(
      renderState = RenderState(None, None, initialWidth, initialHeight),
      hasSaved = false,
      needsRender = true
    )
  )

  private lazy val batch = new SpriteBatch()

  def markNeedsRender(): Unit =
    state.updateAndGet(s => s.copy(needsRender = true))

  def markSaved(): Unit =
    state.updateAndGet(s => s.copy(hasSaved = true))

  def hasSaved: Boolean = state.get().hasSaved

  def needsRender: Boolean = state.get().needsRender

  def currentDimensions: (Int, Int) =
    val rs = state.get().renderState
    (rs.width, rs.height)

  def renderToScreen(rgbaBytes: Array[Byte], width: Int, height: Int): Unit =
    val updated = state.updateAndGet { current =>
      val newRenderState = updateRenderState(current.renderState, rgbaBytes, width, height)
      current.copy(renderState = newRenderState, needsRender = false)
    }

    // Draw using the updated state (side effect separated from state update)
    updated.renderState.texture.foreach { tex =>
      batch.begin()
      batch.draw(tex, 0, 0, width.toFloat, height.toFloat)
      batch.end()
    }

  def redrawExisting(width: Int, height: Int): Unit =
    state.get().renderState.texture.foreach { tex =>
      batch.begin()
      batch.draw(tex, 0, 0, width.toFloat, height.toFloat)
      batch.end()
    }

  def dispose(): Unit =
    state.get().renderState.dispose()
    batch.dispose()

  private def updateRenderState(
    current: RenderState,
    rgbaBytes: Array[Byte],
    width: Int,
    height: Int
  ): RenderState =
    // Pure state transition
    val resized = if width != current.width || height != current.height then
      current.dispose()
      RenderState(None, None, width, height)
    else current

    val (withPixmap, pm) = resized.ensurePixmap()
    withPixmap.updatePixmap(rgbaBytes, pm)
    val (withTexture, _) = withPixmap.ensureTexture(pm)
    withTexture
