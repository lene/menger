package menger.engines

import io.github.lene.optix.OptiXException
import io.github.lene.optix.TextureUploadException
import menger.common.EnvironmentException
import menger.common.InternalException
import menger.common.MengerException
import menger.engines.scene.AssetPaths
import menger.scene.LoadError

/** Why a window could not show a scene, as its seams (a frame that failed to build, a reload that
  * failed, a first scene that failed) report it. The one place an exception is classified
  * (menger#103): the scene's content was refused, the environment failed, or it is a bug. */
enum SceneFailure:
  /** The scene file did not load. */
  case Load(error: LoadError)
  /** The scene's content was refused: over budget, a failed precondition, an unknown option. */
  case Invalid(detail: String)
  /** The GPU, the native layer or the disk failed; the scene may be fine. */
  case Environment(detail: String, error: Throwable)
  /** A bug: nothing the scene's author can fix. */
  case Internal(detail: String, error: Throwable)

  def kind: String = this match
    case _: Load        => "load"
    case _: Invalid     => "scene"
    case _: Environment => "environment"
    case _: Internal    => "internal"

  def message: String = this match
    case Load(error)           => error.message
    case Invalid(detail)       => detail
    case Environment(detail, _) => detail
    case Internal(detail, _)    => detail

  /** A bug, as opposed to the scene's fault or the environment's: what deserves a stack trace in
    * front of a person. */
  def isBug: Boolean = this match
    case _: Internal => true
    case _           => false

  /** The throwable to log a stack trace for: only a failure that is not the scene's. */
  def cause: Option[Throwable] = this match
    case Environment(_, error) => Some(error)
    case Internal(_, error)    => Some(error)
    case _                     => None

  /** What a person or the agent reads: the message, labelled with the kind unless it is a load
    * error, whose text already says what failed. */
  def describe: String = this match
    case _: Load => message
    case _       => s"[$kind] $message"

  /** The same failure with the step that failed in front of its message. */
  def prefixed(step: String): SceneFailure = this match
    case load: Load       => load
    case Invalid(detail)  => Invalid(s"$step: $detail")
    case Environment(detail, error) => Environment(s"$step: $detail", error)
    case Internal(detail, error)    => Internal(s"$step: $detail", error)

object SceneFailure:

  /** Classifies by what was thrown. The typed exceptions say it themselves (`Fail`); the rest is
    * a heuristic on the exception class, as good as the throw sites that have not been typed. */
  def of(cause: Throwable): SceneFailure =
    val detail = Option(cause.getMessage).getOrElse(cause.getClass.getSimpleName)
    cause match
      case _: EnvironmentException => Environment(detail, cause)
      case _: InternalException    => Internal(detail, cause)
      case _: MengerException | _: IllegalArgumentException | _: UnsupportedOperationException |
           _: AssetPaths.AssetPathException => Invalid(detail)
      case _: OptiXException | _: TextureUploadException | _: java.io.IOException =>
        Environment(detail, cause)
      case _ => Internal(detail, cause)
