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

  protected def drawables: List[ModelInstance] =
    given ProfilingConfig = profilingConfig
    val currentLevel = currentAnimatedLevel
    generateObjectWithOverlay(spongeType, currentLevel).get.getModel  // Throws on failure - caught by Main

  private def currentAnimatedLevel: Float =
    val currentFrame = frameCounter.get()
    animationSpecifications.level(currentFrame).getOrElse(spongeLevel)

  protected def gdxResources: GDXResources = GDXResources(None, fpsLogIntervalMs)

  override def currentRotProj: RotationProjectionParameters =
    val currentFrame = frameCounter.get()
    animationSpecifications.rotationProjectionParameters(currentFrame).get + rotationProjectionParameters  // Throws on failure - caught by Main

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
