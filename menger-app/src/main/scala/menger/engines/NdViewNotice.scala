package menger.engines

import java.util.concurrent.atomic.AtomicBoolean

import com.typesafe.scalalogging.LazyLogging
import menger.spec.ObjectSpec

/** The one log line for what the window cannot do to an object with a `ProjectionSpec`
  * (menger#162): its Shift+drag rotation is a 4D rotation. */
private[engines] object NdViewNotice extends LazyLogging:

  private val shown = AtomicBoolean(false)

  def rotationIgnored(spec: ObjectSpec): Unit =
    if shown.compareAndSet(false, true) then
      logger.info(
        s"Shift+drag does not rotate ${spec.objectType}: it has a ProjectionSpec, whose " +
          "rotation is set by the scene"
      )
