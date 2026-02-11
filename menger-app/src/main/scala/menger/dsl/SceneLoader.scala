package menger.dsl

import scala.util.Try

import com.typesafe.scalalogging.LazyLogging

/** Loader for pre-compiled DSL scenes.
  *
  * Supports two loading mechanisms:
  * 1. Registry lookup: Simple names like "glass-sphere" registered via SceneRegistry
  * 2. Reflection: Fully-qualified class names like "examples.dsl.GlassSphere"
  *
  * Example scene object:
  * ```scala
  * package examples.dsl
  * import menger.dsl.*
  *
  * object MyScene:
  *   val scene = Scene(
  *     camera = Camera((0f, 0f, 3f), (0f, 0f, 0f)),
  *     objects = List(Sphere(Material.Glass)),
  *     lights = List(Directional((1f, -1f, -1f)))
  *   )
  *
  *   // Optional: Register for short name access
  *   SceneRegistry.register("my-scene", scene)
  * ```
  *
  * Usage:
  * - `--scene glass-sphere` (registry lookup)
  * - `--scene examples.dsl.GlassSphere` (reflection)
  */
object SceneLoader extends LazyLogging:

  /** Load a scene by name or fully-qualified class name.
    *
    * First attempts registry lookup, then tries reflection if that fails.
    *
    * @param sceneName Simple name or fully-qualified class name
    * @return Either an error message or the loaded Scene
    */
  def load(sceneName: String): Either[String, Scene] =
    logger.info(s"Loading scene: $sceneName")

    // Try registry first
    SceneRegistry.get(sceneName) match
      case Some(scene) =>
        logger.info(s"Loaded scene from registry: $sceneName")
        Right(scene)
      case None =>
        // Try reflection
        loadByReflection(sceneName)

  /** Load a scene by fully-qualified class name using reflection.
    *
    * Expects an object with a `scene` field of type Scene.
    *
    * @param className Fully-qualified class name (e.g., "examples.dsl.GlassSphere")
    * @return Either an error message or the loaded Scene
    */
  private def loadByReflection(className: String): Either[String, Scene] =
    logger.debug(s"Attempting to load scene via reflection: $className")

    Try {
      // Get the class
      val cls = Class.forName(s"$className$$")

      // Get the MODULE$ field (Scala object singleton)
      val moduleField = cls.getDeclaredField("MODULE$")
      moduleField.setAccessible(true)
      // scalafix:off DisableSyntax.null
      // Note: null is correct here - getting static field via Java reflection
      val module = moduleField.get(null)
      // scalafix:on DisableSyntax.null

      // Get the scene field
      val sceneField = cls.getDeclaredField("scene")
      sceneField.setAccessible(true)
      val scene = sceneField.get(module)

      scene match
        case s: Scene => Right(s)
        case _ => Left(s"Object '$className' has a 'scene' field but it's not of type Scene")
    }.toEither match
      case Right(result) =>
        logger.info(s"Successfully loaded scene via reflection: $className")
        result
      case Left(ex: ClassNotFoundException) =>
        Left(s"Scene not found: '$className'. Available registered scenes: ${SceneRegistry.list().mkString(", ")}")
      case Left(ex: NoSuchFieldException) =>
        Left(s"Object '$className' does not have a 'scene' field of type Scene")
      case Left(ex) =>
        Left(s"Failed to load scene '$className': ${ex.getMessage}")

  /** List all available scenes (from registry only). */
  def listAvailable(): List[String] =
    SceneRegistry.list()
