package menger.dsl

import scala.util.Try

import com.typesafe.scalalogging.LazyLogging

/** Loader for pre-compiled DSL scenes.
  *
  * Supports two loading mechanisms:
  * 1. Registry lookup: Simple names like "glass-sphere" registered via SceneRegistry
  * 2. Reflection: Fully-qualified class names like "examples.dsl.GlassSphere"
  *
  * Detects both static scenes (`val scene: Scene`) and animated scenes
  * (`def scene(t: Float): Scene`). Returns a [[LoadedScene]] ADT.
  *
  * Example static scene object:
  * ```scala
  * object MyScene:
  *   val scene = Scene(
  *     camera = Camera((0f, 0f, 3f), (0f, 0f, 0f)),
  *     objects = List(Sphere(Material.Glass)),
  *     lights = List(Directional((1f, -1f, -1f)))
  *   )
  * ```
  *
  * Example animated scene object:
  * ```scala
  * object OrbitingSphere:
  *   def scene(t: Float): Scene = Scene(...)
  * ```
  *
  * Usage:
  * - `--scene glass-sphere` (registry lookup)
  * - `--scene examples.dsl.GlassSphere` (reflection)
  */
object SceneLoader extends LazyLogging:

  /** Load a scene by name or fully-qualified class name.
    *
    * First attempts registry lookup (always returns Static), then tries reflection.
    *
    * @param sceneName Simple name or fully-qualified class name
    * @return Either an error message or the loaded LoadedScene
    */
  def load(sceneName: String): Either[String, LoadedScene] =
    logger.info(s"Loading scene: $sceneName")

    // Try registry first (registry only stores static scenes)
    SceneRegistry.get(sceneName) match
      case Some(scene) =>
        logger.info(s"Loaded scene from registry: $sceneName")
        Right(LoadedScene.Static(scene))
      case None =>
        // Try reflection
        loadByReflection(sceneName)

  /** Load a scene by fully-qualified class name using reflection.
    *
    * First tries `val scene: Scene` (static), then tries `def scene(t: Float): Scene` (animated).
    *
    * @param className Fully-qualified class name (e.g., "examples.dsl.GlassSphere")
    * @return Either an error message or the loaded LoadedScene
    */
  private def loadByReflection(className: String): Either[String, LoadedScene] =
    logger.debug(s"Attempting to load scene via reflection: $className")

    Try {
      val cls = Class.forName(s"$className$$")
      val moduleField = cls.getDeclaredField("MODULE$")
      moduleField.setAccessible(true)
      // scalafix:off DisableSyntax.null
      // Note: null is correct here - getting static field via Java reflection
      val module = moduleField.get(null)
      // scalafix:on DisableSyntax.null

      // Try static field first
      tryLoadStaticScene(cls, module)
        .orElse(tryLoadAnimatedScene(cls, module))
        .getOrElse(Left(s"Object '$className' has neither a 'scene' field of type Scene " +
          "nor a 'scene(Float)' method returning Scene"))
    }.toEither match
      case Right(result) =>
        logger.info(s"Successfully loaded scene via reflection: $className")
        result
      case Left(ex: ClassNotFoundException) =>
        val registered = SceneRegistry.list()
        val hint = if registered.nonEmpty then s"Registered short names: ${registered.mkString(", ")}"
                   else "Use a fully-qualified class name (e.g. 'examples.dsl.GlassSphere')"
        Left(s"Scene not found: '$className'. $hint")
      case Left(ex) =>
        Left(s"Failed to load scene '$className': ${ex.getMessage}")

  private def tryLoadStaticScene(cls: Class[?], module: AnyRef): Option[Either[String, LoadedScene]] =
    Try {
      val sceneField = cls.getDeclaredField("scene")
      sceneField.setAccessible(true)
      sceneField.get(module) match
        case s: Scene => Right(LoadedScene.Static(s))
        case other => Left(s"'scene' field exists but is ${other.getClass.getName}, not Scene")
    }.toOption

  private def tryLoadAnimatedScene(cls: Class[?], module: AnyRef): Option[Either[String, LoadedScene]] =
    Try {
      val sceneMethod = cls.getDeclaredMethod("scene", java.lang.Float.TYPE)
      // Verify the method returns a Scene by calling it with t=0
      sceneMethod.invoke(module, java.lang.Float.valueOf(0f)) match
        case _: Scene =>
          val fn: Float => Scene = t => invokeSceneMethod(sceneMethod, module, t)
          Right(LoadedScene.Animated(fn))
        case other =>
          Left(s"'scene(Float)' method exists but returns ${other.getClass.getName}, not Scene")
    }.toOption

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  private def invokeSceneMethod(method: java.lang.reflect.Method, module: AnyRef, t: Float): Scene =
    method.invoke(module, java.lang.Float.valueOf(t)) match
      case s: Scene => s
      case other => throw IllegalStateException(
        s"scene(Float) unexpectedly returned ${other.getClass.getName} instead of Scene"
      )

  /** List all available scenes (from registry only). */
  def listAvailable(): List[String] =
    SceneRegistry.list()
