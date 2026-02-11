package menger.dsl

import scala.collection.mutable

/** Registry for pre-compiled DSL scenes.
  *
  * Scenes can register themselves to make them available via CLI.
  * Usage: SceneRegistry.register("my-scene", myScene)
  */
object SceneRegistry:
  private val scenes = mutable.Map[String, Scene]()

  /** Register a scene with a name for CLI access.
    *
    * @param name Scene identifier (e.g., "glass-sphere", "menger-showcase")
    * @param scene The Scene instance to register
    */
  def register(name: String, scene: Scene): Unit =
    scenes(name) = scene

  /** Get a scene by name.
    *
    * @param name Scene identifier
    * @return Some(scene) if found, None otherwise
    */
  def get(name: String): Option[Scene] =
    scenes.get(name)

  /** List all registered scene names. */
  def list(): List[String] =
    scenes.keys.toList.sorted

  /** Check if a scene is registered. */
  def contains(name: String): Boolean =
    scenes.contains(name)

  /** Clear all registered scenes (primarily for testing). */
  def clear(): Unit =
    scenes.clear()
