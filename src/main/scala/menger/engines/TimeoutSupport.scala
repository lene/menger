package menger.engines

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.utils.Timer
import com.typesafe.scalalogging.LazyLogging

trait TimeoutSupport extends LazyLogging:
  def timeout: Float

  protected def startExitTimer(timeoutSeconds: Float): Unit =
    if timeoutSeconds > 0 then
      logger.debug(s"Starting timer for $timeoutSeconds seconds")
      Timer.schedule(() => Gdx.app.exit(), timeoutSeconds, 0)
