package menger.scene

/** Why a scene could not be loaded. The cases are what callers branch on (`SceneValidator`
  * picks its result tag by them, the window reports them as `reload_failed`); [[message]] is the
  * text a person reads. */
enum LoadError:
  case NotFound(path: String)
  case UnknownScene(name: String, registered: List[String])
  case NoSceneObject(fileName: String)
  case CompileFailed(fileName: String, diagnostics: List[LoadError.Diagnostic])
  case ConstructionFailed(detail: String)
  case StackOverflow(detail: String)
  case ClasspathBroken(detail: String)
  case Malformed(detail: String)

  /** Stable machine-readable name of the case; part of the validator's result contract. */
  def kind: String = this match
    case _: NotFound           => "not_found"
    case _: UnknownScene       => "unknown_scene"
    case _: NoSceneObject      => "no_scene_object"
    case _: CompileFailed      => "compile_failed"
    case _: ConstructionFailed => "construction_failed"
    case _: StackOverflow      => "stack_overflow"
    case _: ClasspathBroken    => "classpath_broken"
    case _: Malformed          => "malformed"

  def message: String = this match
    case NotFound(path) => s"Scene file not found: $path"
    case UnknownScene(name, registered) =>
      val hint =
        if registered.nonEmpty then s"Registered short names: ${registered.mkString(", ")}"
        else "Use a fully-qualified class name (e.g. 'examples.dsl.GlassSphere')"
      s"Scene not found: '$name'. $hint"
    case NoSceneObject(fileName) =>
      s"Could not find a top-level 'object' declaration in '$fileName'"
    case CompileFailed(fileName, diagnostics) =>
      s"Compilation of '$fileName' failed:\n${diagnostics.map(_.text).mkString("\n")}"
    case ConstructionFailed(detail) => s"scene construction failed: $detail"
    case StackOverflow(detail) =>
      s"scene construction overflowed the stack (infinite recursion?): $detail"
    case ClasspathBroken(detail) => s"scene classpath is broken: $detail"
    case Malformed(detail) => detail

object LoadError:
  /** One compiler error; `line` is 1-based and absent when the compiler gave no position. */
  case class Diagnostic(line: Option[Int], message: String):
    def text: String = line.fold(message)(l => s"line $l: $message")
