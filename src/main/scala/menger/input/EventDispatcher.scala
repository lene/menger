package menger.input

import menger.RotationProjectionParameters

trait Observer:
  def handleEvent(event: RotationProjectionParameters): Unit


class EventDispatcher(observers: List[Observer] = Nil):
  def this() = this(Nil)

  def withObserver(observer: Observer): EventDispatcher =
    EventDispatcher(observer :: observers)

  def notifyObservers(event: RotationProjectionParameters): Unit =
    observers.foreach(_.handleEvent(event))
