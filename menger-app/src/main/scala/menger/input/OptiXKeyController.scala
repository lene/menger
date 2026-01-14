package menger.input

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.Input.Keys
import com.typesafe.scalalogging.LazyLogging
import menger.RotationProjectionParameters

class OptiXKeyController(dispatcher: EventDispatcher) extends BaseKeyController with LazyLogging:

  private final val rotateAngle = 45f

  def update(): Unit =
    if rotatePressed.values.exists(_ == true) then
      val delta = Gdx.graphics.getDeltaTime
      if shift then
        logger.debug(s"Shift pressed with rotation keys, delta=$delta")
        onShiftPressed(delta)

  override protected def handleEscape(): Boolean =
    Gdx.app.exit()
    true

  override protected def onRotationUpdate(): Unit = update()

  private def onShiftPressed(delta: Float): Unit =
    val rotXW = angle(delta, Seq(Keys.LEFT, Keys.RIGHT))
    val rotYW = angle(delta, Seq(Keys.UP, Keys.DOWN))
    val rotZW = angle(delta, Seq(Keys.PAGE_UP, Keys.PAGE_DOWN))
    logger.debug(s"Dispatching 4D rotation event: rotXW=$rotXW, rotYW=$rotYW, rotZW=$rotZW")
    dispatcher.notifyObservers(
      RotationProjectionParameters(rotXW, rotYW, rotZW)
    )

  private val factor = Map(
    Keys.RIGHT -> -1, Keys.LEFT -> 1, Keys.UP -> 1, Keys.DOWN -> -1,
    Keys.PAGE_UP -> 1, Keys.PAGE_DOWN -> -1
  )
  private def angle(delta: Float, keys: Seq[Int]): Float = delta * rotateAngle * direction(keys)
  private def direction(keys: Seq[Int]) = keys.find(rotatePressed).map(factor(_)).getOrElse(0)
