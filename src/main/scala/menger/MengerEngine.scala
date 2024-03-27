package menger

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.{Material, ModelInstance}
import com.badlogic.gdx.math.Vector3
import com.badlogic.gdx.utils.Timer
import com.badlogic.gdx.{Game, Gdx}
import menger.input.EventDispatcher
import menger.objects.higher_d.{RotatedProjection, Tesseract, TesseractSponge, TesseractSponge2}
import menger.objects.{Builder, Geometry, SpongeBySurface, SpongeByVolume}

class MengerEngine(
  timeout: Float = 0, spongeLevel: Int = 0, lines: Boolean = false, spongeType: String = "square",
  rotationProjectionParameters: RotationProjectionParameters = RotationProjectionParameters()
) extends Game:

  private val material = Builder.WHITE_MATERIAL
  private lazy val primitiveType = if lines then GL20.GL_LINES else GL20.GL_TRIANGLES
  private lazy val sponge: Geometry = generateObject(spongeType, spongeLevel, material, primitiveType)
  private lazy val eventDispatcher = EventDispatcher()
  "tesseract".r.findFirstIn(spongeType).foreach(_ => eventDispatcher.addObserver(sponge))
  private def drawables: List[ModelInstance] = sponge.at(Vector3(0, 0, 0), 1)
  private lazy val gdxResources = GDXResources(eventDispatcher)

  override def create(): Unit =
    Gdx.app.log("create", sponge.toString())
    if timeout > 0 then
      Timer.schedule(() => Gdx.app.exit(), timeout, 0)

  override def render(): Unit = gdxResources.render(drawables)

  override def dispose(): Unit = gdxResources.dispose()

  override def resume(): Unit = {}

  override def resize(width: Int, height: Int): Unit = gdxResources.resize()

  override def pause(): Unit = {}

  private def generateObject(spongeType: String, level: Int, material: Material, primitiveType: Int): Geometry =
    spongeType match
      case "square" => SpongeBySurface(level, material, primitiveType)
      case "cube" => SpongeByVolume(level, material, primitiveType)
      case "tesseract" => RotatedProjection(
        Tesseract(), rotationProjectionParameters, material, primitiveType
      )
      case "tesseract-sponge" => RotatedProjection(
        TesseractSponge(level), rotationProjectionParameters, material, primitiveType
      )
      case "tesseract-sponge-2" => RotatedProjection(
        TesseractSponge2(level), rotationProjectionParameters, material, primitiveType
      )
      case _ => throw new IllegalArgumentException(s"Unknown sponge type: $spongeType")
