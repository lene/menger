package menger.engines

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.typesafe.scalalogging.LazyLogging
import menger.AnimationSpecifications
import menger.GDXResources
import menger.ProfilingConfig
import menger.RotationProjectionParameters
import menger.common.Const

class AnimatedMengerEngine(
  spongeType: String, spongeLevel: Float,
  rotationProjectionParameters: RotationProjectionParameters = RotationProjectionParameters(),
  lines: Boolean, color: Color, val animationSpecifications: AnimationSpecifications,
  val saveName: Option[String], faceColor: Option[Color] = None, lineColor: Option[Color] = None,
  fpsLogIntervalMs: Int = Const.fpsLogIntervalMs
)(using config: ProfilingConfig) extends MengerEngine(spongeType, spongeLevel, rotationProjectionParameters, lines, color, faceColor, lineColor, fpsLogIntervalMs)
with LazyLogging with SavesScreenshots:
  private val frameCounter = java.util.concurrent.atomic.AtomicInteger(0)

  // Helper to unwrap Try with graceful error handling and app exit on failure
  private def unwrapOrExit[A](t: scala.util.Try[A], errorContext: String): A =
    t.getOrElse {
      t.failed.foreach { e =>
        logger.error(s"$errorContext: ${e.getMessage}", e)
        Gdx.app.exit()
      }
      ??? // Control never reaches here - app exits above
    }

  protected def drawables: List[ModelInstance] =
    given ProfilingConfig = profilingConfig
    val currentLevel = currentAnimatedLevel
    val geometry = unwrapOrExit(
      generateObjectWithOverlay(spongeType, currentLevel),
      s"Failed to generate object for frame ${frameCounter.get()}"
    )
    geometry.getModel

  private def currentAnimatedLevel: Float =
    val currentFrame = frameCounter.get()
    animationSpecifications.level(currentFrame).getOrElse(spongeLevel)

  protected def gdxResources: GDXResources = GDXResources(None, fpsLogIntervalMs)

  override def currentRotProj: RotationProjectionParameters =
    val currentFrame = frameCounter.get()
    val params = unwrapOrExit(
      animationSpecifications.rotationProjectionParameters(currentFrame),
      s"Failed to get rotation parameters for frame $currentFrame"
    )
    params + rotationProjectionParameters

  override def create(): Unit =
    logger.info(s"Animating for $animationSpecifications")

  override def render(): Unit =
    val currentFrame = frameCounter.get()
    logger.info(s"frame: $currentFrame |$currentRotProj| ${currentSaveName.getOrElse("")}")
    super.render()
    gdxResources.render(drawables)
    saveImage()
    nextStep()

  private def nextStep(): Unit =
    val nextFrame = frameCounter.incrementAndGet()
    if nextFrame >= animationSpecifications.numFrames then
      Gdx.app.exit()

  protected def currentSaveName: Option[String] = saveName.map(_.format(frameCounter.get()))
