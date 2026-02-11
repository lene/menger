package menger.dsl

import scala.io.Source
import scala.util.{Try, Using}
import com.typesafe.scalalogging.LazyLogging

/** Loader for scene definition files written in Scala DSL.
  *
  * Scene files are Scala source files that define a scene using the DSL API.
  * The file must evaluate to a Scene value.
  *
  * Example scene file:
  * ```
  * import menger.dsl.*
  *
  * Scene(
  *   camera = Camera((0f, 0f, 3f), (0f, 0f, 0f)),
  *   objects = List(
  *     Sphere((0f, 0f, 0f), Material.Glass),
  *     Cube((2f, 0f, 0f), Material.Gold)
  *   ),
  *   lights = List(
  *     Directional((1f, -1f, -1f))
  *   )
  * )
  * ```
  */
object SceneLoader extends LazyLogging:

  /** Load and evaluate a scene definition from a file.
    *
    * @param path Path to the scene definition file
    * @return Either an error message or the loaded Scene
    */
  def loadFromFile(path: String): Either[String, Scene] =
    logger.info(s"Loading scene from file: $path")

    // Read file content
    val contentResult = Using(Source.fromFile(path)) { source =>
      source.mkString
    }.toEither.left.map { ex =>
      s"Failed to read scene file '$path': ${ex.getMessage}"
    }

    contentResult.flatMap { content =>
      logger.debug(s"File content loaded, ${content.length} characters")
      parseScene(content, path)
    }

  /** Parse and evaluate scene definition from a string.
    *
    * @param content Scene definition as Scala code
    * @param sourceName Name for error messages (typically filename)
    * @return Either an error message or the parsed Scene
    */
  def parseScene(content: String, sourceName: String = "<inline>"): Either[String, Scene] =
    Try {
      // Use Scala's reflection/compilation API to evaluate the DSL
      // For now, return an error as this requires runtime compilation
      Left(s"Scene loading from file not yet implemented. Scene files must be pre-compiled.")
    }.toEither match
      case Right(result) => result
      case Left(ex) => Left(s"Failed to parse scene from '$sourceName': ${ex.getMessage}")

  /** Validate that a scene definition is syntactically correct.
    *
    * @param content Scene definition as Scala code
    * @return Either validation errors or Unit on success
    */
  def validate(content: String): Either[String, Unit] =
    parseScene(content, "<validation>").map(_ => ())
