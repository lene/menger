package menger.input

import menger.RotationProjectionParameters

trait Observer:
  def handleEvent(event: RotationProjectionParameters): Unit


class EventDispatcher:
  private var observers: List[Observer] = Nil

  def addObserver(observer: Observer): Unit =
    observers = observer :: observers

  def withObserver(observer: Observer): EventDispatcher =
    addObserver(observer)
    this

  def notifyObservers(event: RotationProjectionParameters): Unit =
    observers.foreach(_.handleEvent(event))
