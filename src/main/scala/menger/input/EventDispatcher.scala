package menger.input

import java.util.concurrent.atomic.AtomicReference

import menger.RotationProjectionParameters

trait Observer:
  def handleEvent(event: RotationProjectionParameters): Unit


class EventDispatcher:
  private val observers = new AtomicReference[List[Observer]](Nil)

  def addObserver(observer: Observer): Unit =
    observers.updateAndGet(observer :: _)

  def withObserver(observer: Observer): EventDispatcher =
    addObserver(observer)
    this

  def notifyObservers(event: RotationProjectionParameters): Unit =
    observers.get().foreach(_.handleEvent(event))
