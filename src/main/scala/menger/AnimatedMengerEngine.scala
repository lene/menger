package menger

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.math.Vector3

class AnimatedMengerEngine(
  spongeType: String, spongeLevel: Int,
  rotationProjectionParameters: RotationProjectionParameters = RotationProjectionParameters(),
  lines: Boolean, val animationSpecifications: AnimationSpecifications
) extends MengerEngine(spongeType, spongeLevel, rotationProjectionParameters, lines):
  private var currentFrame: Int = 0

  protected def drawables: List[ModelInstance] = 
    generateObject(spongeType, spongeLevel, material, primitiveType).at(Vector3(0, 0, 0), 1)

  protected def gdxResources: GDXResources = GDXResources(None)

  override def currentRotProj: RotationProjectionParameters =
    val r = rotationProjectionParameters + animationSpecifications.rotationProjectionParameters(currentFrame)
    Gdx.app.log(s"${getClass.getSimpleName}.currentRotProj()", s"frame: $currentFrame $r")
    r

  override def create(): Unit =
    Gdx.app.log(s"${getClass.getSimpleName}.create()", s"Animating for $animationSpecifications")

  override def render(): Unit =
    super.render()
    gdxResources.render(drawables)
    nextStep()
    

  private def nextStep(): Unit = 
    currentFrame += 1
    if currentFrame >= animationSpecifications.numFrames then
      Gdx.app.exit()
