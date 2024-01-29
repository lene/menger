package menger

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.{GL20, PerspectiveCamera}
import com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute
import com.badlogic.gdx.graphics.g3d.environment.DirectionalLight
import com.badlogic.gdx.graphics.g3d.{Environment, ModelBatch, RenderableProvider}
import com.badlogic.gdx.math.Vector3

import scala.collection.immutable.List
import scala.jdk.CollectionConverters.*


case class GDXResources():

  private val cameraPosition = Vector3(-2f, 1f, -1f)
  private val environment: Environment = createEnvironment
  private val camera: PerspectiveCamera = createCamera(cameraPosition)
  Gdx.input.setInputProcessor(MengerInputMultiplexer(camera))
  private val modelBatch = ModelBatch()

  def render(models: List[RenderableProvider]*): Unit =
    Gdx.gl.glViewport(0, 0, Gdx.graphics.getWidth, Gdx.graphics.getHeight)
    Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)
    modelBatch.begin(camera)
    modelBatch.render(models.flatten.asJava, environment)
    modelBatch.end()


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
