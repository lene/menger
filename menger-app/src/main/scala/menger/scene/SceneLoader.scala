package menger.scene

import scala.util.Try

import com.typesafe.scalalogging.LazyLogging
import menger.common.ConfigurationException
import menger.dsl.LoadedScene
import menger.dsl.Scene
import menger.dsl.SceneRegistry

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
    * @return Either why it could not be loaded or the loaded LoadedScene
    */
  def load(sceneName: String): Either[LoadError, LoadedScene] =
    logger.info(s"Loading scene: $sceneName")
    // Total (review 2026-10-14, menger#101): a failing `require` in a scene's `val scene = ...`
    // surfaces as ExceptionInInitializerError, a LinkageError that neither `Try` nor a
    // `catch Exception` sees -- live reload silently did nothing and the first load ended in a
    // stack trace. The three Errors a scene can provoke become a `Left` here (`ClasspathBroken`
    // is the environment's, the others are scene defects).
    try loadUnguarded(sceneName)
    catch
      case e: ExceptionInInitializerError =>
        Left(LoadError.ConstructionFailed(
          Option(e.getCause).map(_.getMessage).getOrElse(e.getMessage)
        ))
      case e: StackOverflowError => Left(LoadError.StackOverflow(e.toString))
      case e: LinkageError => Left(LoadError.ClasspathBroken(e.toString))

  private def loadUnguarded(sceneName: String): Either[LoadError, LoadedScene] =
    // A `.scala` name is a file, found or not: it never falls through to `Class.forName`,
    // which would initialise whatever class the name happens to spell (review 2026-10-14,
    // menger#88).
    if sceneName.endsWith(".scala") then
      loadFromFile(new java.io.File(sceneName))
    else
      // Try registry first (registry only stores static scenes)
      SceneRegistry.get(sceneName) match
        case Some(scene) =>
          logger.info(s"Loaded scene from registry: $sceneName")
          Right(LoadedScene.Static(scene))
        case None =>
          loadObject(sceneName, getClass.getClassLoader,
            LoadError.UnknownScene(sceneName, SceneRegistry.list()))

  private def loadFromFile(file: java.io.File): Either[LoadError, LoadedScene] =
    if !file.isFile then Left(LoadError.NotFound(file.getAbsolutePath))
    else SceneCompiler.compile(file).flatMap { loader =>
      detectObjectName(file) match
        case None => Left(LoadError.NoSceneObject(file.getName))
        case Some((pkg, objName)) =>
          val className = if pkg.isEmpty then objName else s"$pkg.$objName"
          loadObject(className, loader,
            LoadError.Malformed(s"Failed to load scene '$className': class not found"))
    }

  private def detectObjectName(file: java.io.File): Option[(String, String)] =
    val lines = scala.io.Source.fromFile(file).getLines().take(60).toSeq
    val pkg = lines.collectFirst {
      case l if l.trim.startsWith("package ") => l.trim.stripPrefix("package ").trim
    }.getOrElse("")
    val obj = lines.collectFirst {
      case l if l.trim.startsWith("object ") =>
        l.trim.stripPrefix("object ").takeWhile(c => c.isLetterOrDigit || c == '_')
    }
    obj.map(o => (pkg, o))

  /** Loads the Scala object `className` through `loader` and reads its scene: first
    * `val scene: Scene` (static), then `def scene(t: Float): Scene` (animated). `notFound` is
    * what a missing class means to the caller: an unknown name, or a compiled file whose object
    * the loader cannot see. */
  private def loadObject(
    className: String,
    loader: ClassLoader,
    notFound: => LoadError
  ): Either[LoadError, LoadedScene] =
    logger.debug(s"Attempting to load scene via reflection: $className")
    Try {
      val cls = loader.loadClass(s"$className$$")
      val moduleField = cls.getDeclaredField("MODULE$")
      moduleField.setAccessible(true)
      // scalafix:off DisableSyntax.null
      // Note: null is correct here - getting static field via Java reflection
      val module = moduleField.get(null)
      // scalafix:on DisableSyntax.null
      tryLoadStaticScene(cls, module)
        .orElse(tryLoadAnimatedScene(cls, module))
        .getOrElse(Left(LoadError.Malformed(
          s"Object '$className' has neither a 'scene' field of type Scene " +
          "nor a 'scene(Float)' method returning Scene")))
    }.toEither match
      case Right(result) =>
        logger.info(s"Successfully loaded scene via reflection: $className")
        result
      case Left(_: ClassNotFoundException) => Left(notFound)
      case Left(ex) =>
        Left(LoadError.Malformed(s"Failed to load scene '$className': ${ex.getMessage}"))

  private def tryLoadStaticScene(
    cls: Class[?], module: AnyRef
  ): Option[Either[LoadError, LoadedScene]] =
    Try {
      val sceneField = cls.getDeclaredField("scene")
      sceneField.setAccessible(true)
      sceneField.get(module) match
        case s: Scene => Right(LoadedScene.Static(s))
        case other => Left(LoadError.Malformed(
          s"'scene' field exists but is ${other.getClass.getName}, not Scene"))
    }.toOption

  private def tryLoadAnimatedScene(
    cls: Class[?], module: AnyRef
  ): Option[Either[LoadError, LoadedScene]] =
    // Only a missing method means "not an animated scene". An exception thrown while probing
    // scene(0) is the scene's own error and is reported as such -- swallowing it used to turn
    // e.g. a failed require() into a misleading "has no scene method".
    Try(cls.getDeclaredMethod("scene", java.lang.Float.TYPE)).toOption.map { sceneMethod =>
      Try(sceneMethod.invoke(module, java.lang.Float.valueOf(0f))).toEither match
        case Left(e) =>
          Left(LoadError.Malformed(s"scene(0) threw: ${rootCause(e).getMessage}"))
        case Right(_: Scene) =>
          loadDuration(cls, module).map { duration =>
            val fn: Float => Scene = t => invokeSceneMethod(sceneMethod, module, t)
            LoadedScene.Animated(fn)(duration)
          }
        case Right(other) =>
          Left(LoadError.Malformed(
            s"'scene(Float)' method exists but returns ${other.getClass.getName}, not Scene"))
    }

  /** The scene object's optional `val durationSeconds: Float`, read through its accessor. The
    * old name `duration` did not say it is seconds (usability review 2026-10, session 3, F84);
    * it is still read, with a deprecation warning, when `durationSeconds` is absent. */
  private def loadDuration(cls: Class[?], module: AnyRef): Either[LoadError, Option[Float]] =
    val accessors = List("durationSeconds", "duration").flatMap(name =>
      Try(cls.getDeclaredMethod(name)).toOption.map(name -> _)
    )
    accessors.headOption match
      case None => Right(None)
      case Some((name, accessor)) =>
        if name == "duration" then
          logger.warn("`val duration` is deprecated: name it `val durationSeconds`")
        Try(accessor.invoke(module)).toOption match
          case Some(d: java.lang.Float) if d.floatValue > 0f => Right(Some(d.floatValue))
          case Some(d: java.lang.Float) =>
            Left(LoadError.Malformed(s"'$name' must be positive, got $d"))
          case _ => Left(LoadError.Malformed(
            s"'$name' must be a Float (seconds), e.g. `val durationSeconds = 10f`"))

  @scala.annotation.tailrec
  private def rootCause(e: Throwable): Throwable =
    Option(e.getCause) match
      case Some(cause) if cause ne e => rootCause(cause)
      case _ => e

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  private def invokeSceneMethod(method: java.lang.reflect.Method, module: AnyRef, t: Float): Scene =
    method.invoke(module, java.lang.Float.valueOf(t)) match
      case s: Scene => s
      case other => throw ConfigurationException(
        s"scene(Float) unexpectedly returned ${other.getClass.getName} instead of Scene"
      )

  /** List all available scenes (from registry only). */
  def listAvailable(): List[String] =
    SceneRegistry.list()
