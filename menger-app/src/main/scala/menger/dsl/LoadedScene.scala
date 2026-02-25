package menger.dsl

/** ADT representing a loaded scene: either a static scene or an animated scene parameterized by t. */
sealed trait LoadedScene

object LoadedScene:
  case class Static(scene: Scene) extends LoadedScene
  case class Animated(sceneFunction: Float => Scene) extends LoadedScene
