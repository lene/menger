package menger.failure

import menger.common.ConfigurationException
import menger.common.EnvironmentException
import menger.common.InternalException
import menger.common.NotYetImplementedException
import menger.common.ParseException
import menger.common.UnknownGeometryException

/** The one place production code raises a failure (menger#96, #103). Each method names whose fault
  * it is, and the exception it throws says so to the seams that catch it (`SceneFailure.of`, the
  * exit codes in `Main`): the user's input (`configuration`, `parse`, `unknownGeometry`,
  * `unsupported`), the environment (`environment`) or a bug (`internal`). `sys.error` says none
  * of that: it is a bare RuntimeException, and an ArchUnit rule forbids it. */
object Fail:

  /** The user asked for something that is not valid or not complete. */
  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  def configuration(message: String): Nothing = throw ConfigurationException(message)

  /** A string the program reads (a wire value, an option) is not one of the known forms. */
  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  def parse(message: String, input: String, expected: String): Nothing =
    throw ParseException(message, input, Some(expected))

  /** A geometry type that no builder knows. */
  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  def unknownGeometry(geometryType: String, validTypes: Seq[String]): Nothing =
    throw UnknownGeometryException(geometryType, validTypes)

  /** A valid request for something the program cannot do yet. */
  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  def unsupported(feature: String): Nothing = throw NotYetImplementedException(feature)

  /** Something outside the program is wrong; `remedy` is what the user can do about it. */
  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  def environment(problem: String, remedy: String, cause: Option[Throwable] = None): Nothing =
    throw EnvironmentException(problem, Some(remedy), cause)

  /** As [[environment]], when there is nothing the user can do. */
  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  def environment(problem: String): Nothing = throw EnvironmentException(problem)

  /** An invariant of the program failed: a bug. */
  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  def internal(invariant: String): Nothing = throw InternalException(invariant)
