package menger.dsl

/** ADT representing a loaded scene: either a static scene or an animated scene parameterized by t. */
sealed trait LoadedScene

object LoadedScene:
  case class Static(scene: Scene) extends LoadedScene

  /** `duration` is the scene object's optional `val duration: Float`: t then means seconds in
    * `[0, duration]`, and the interactive window plays the scene in real time, looping. It sits
    * in a second parameter list so `Animated(fn)` patterns keep matching. */
  case class Animated(sceneFunction: Float => Scene)(val duration: Option[Float] = None)
      extends LoadedScene
