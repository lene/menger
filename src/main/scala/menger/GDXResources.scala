package menger

import scala.collection.immutable.List
import scala.jdk.CollectionConverters._

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.PerspectiveCamera
import com.badlogic.gdx.graphics.g3d.Environment
import com.badlogic.gdx.graphics.g3d.ModelBatch
import com.badlogic.gdx.graphics.g3d.RenderableProvider
import com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute
import com.badlogic.gdx.graphics.g3d.environment.DirectionalLight
import com.badlogic.gdx.math.Vector3
import com.typesafe.scalalogging.LazyLogging
import menger.input.EventDispatcher
import menger.input.MengerInputMultiplexer
import org.lwjgl.opengl.GL11


case class GDXResources(eventDispatcher: Option[EventDispatcher]) extends LazyLogging:

  private val cameraPosition = Vector3(-2f, 1f, -1f)
  private val environment: Environment = createEnvironment
  private val camera: PerspectiveCamera = createCamera(cameraPosition)
  eventDispatcher.foreach(e => Gdx.input.setInputProcessor(MengerInputMultiplexer(camera, e)))
  private val modelBatch = ModelBatch()
  private val lastFPSLogTime = java.util.concurrent.atomic.AtomicLong(System.currentTimeMillis())

  def render(models: List[RenderableProvider]*): Unit =
    Gdx.gl.glViewport(0, 0, Gdx.graphics.getWidth, Gdx.graphics.getHeight)
    clear()
    modelBatch.begin(camera)
    modelBatch.render(models.flatten.asJava, environment)
    modelBatch.end()
    logFPS()

  private def logFPS(): Unit =
    if logger.underlying.isDebugEnabled then
      val currentTime = System.currentTimeMillis()
      if currentTime - lastFPSLogTime.get() >= 1000 then
        logger.debug(s"FPS: ${Gdx.graphics.getFramesPerSecond}")
        lastFPSLogTime.set(currentTime)


  def resize(): Unit =
    camera.viewportWidth = Gdx.graphics.getWidth.toFloat
    camera.viewportHeight = Gdx.graphics.getHeight.toFloat
    camera.update()

  def dispose(): Unit = modelBatch.dispose()

  private def createEnvironment: Environment =
    val localEnv = Environment()
    localEnv.set(ColorAttribute(ColorAttribute.AmbientLight, 0.4, 0.4, 0.4, 1.0))
    localEnv.add(DirectionalLight().set(0.8, 0.8, 0.8, -1, -0.8, -0.2))
    localEnv

  private def createCamera(cameraPos: Vector3): PerspectiveCamera =
    val cam = new PerspectiveCamera(67, Gdx.graphics.getWidth.toFloat, Gdx.graphics.getHeight.toFloat) {
      near = 1.0
      far = 300.0
    }
    cam.position.set(cameraPos)
    cam.lookAt(0, 0, 0)
    cam.update()
    cam

  private def clear(): Unit =
    val coverageBit = if Gdx.graphics.getBufferFormat.coverageSampling
    then GL20.GL_COVERAGE_BUFFER_BIT_NV else 0
    Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT | coverageBit)
    Gdx.gl.glHint(GL11.GL_LINE_SMOOTH_HINT, GL11.GL_NICEST)
    Gdx.gl.glEnable(GL11.GL_LINE_SMOOTH)
    Gdx.gl.glHint(GL11.GL_POLYGON_SMOOTH_HINT, GL11.GL_NICEST)
    Gdx.gl.glEnable(GL11.GL_POLYGON_SMOOTH)
