package menger.engines

import scala.util.Try

import com.typesafe.scalalogging.LazyLogging

/** Moving a tracked scene in place (a 4D projection, a pose) instead of rebuilding it. */
object InPlaceMove extends LazyLogging:

  /** The moved state, or None when `move` cannot take the frame or throws. Either way the caller
    * rebuilds: a failed move can leave the scene half moved, and moving it again would build on
    * that (menger#103). */
  def attempt[S](state: S)(move: S => Option[S]): Option[S] =
    val result = Try(move(state))
    result.failed.foreach(e => logger.warn(s"Moving the 4D scene in place failed, rebuilding: $e"))
    result.toOption.flatten
