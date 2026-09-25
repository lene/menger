package menger.tools

import java.io.File

import scala.util.control.NonFatal

import com.typesafe.scalalogging.LazyLogging
import menger.dsl.LoadedScene
import menger.dsl.RestrictedClasspath
import menger.dsl.Scene
import menger.dsl.SceneLoader
import menger.engines.scene.MeshFactory
import menger.objects.higher_d.InvariantFinding
import menger.objects.higher_d.PolytopeInvariants
import upickle.default.ReadWriter
import upickle.default.readwriter
import upickle.default.write

/** Renderer-side gauntlet validation entry point (spec-ai-scene-agent story 5): runs stage 1
  * (compile, via `SceneCompiler` with `RestrictedClasspath` -- AD-4 rule 2) and stage 4
  * (geometric checks -- `SceneLoader.load` already triggers every `require()` precondition at
  * construction time; `PolytopeInvariants` adds the count-independent geometric-invariant
  * check on top) for a single `.scala` scene file, and emits AD-5's tagged result: `ok` |
  * `compile-errors` | `lint-findings` | `refused`.
  *
  * Intentionally out of scope (see the story's `Never` clause): the agent-side static checks
  * (stories 3-4), free-form/lambda object contract testing (`ParametricSurface`/`Curve`/
  * `LSystem`), semantic readback (stage 5, story 6), and wiring this into `menger-scene-
  * agent`'s `generate()`/`revise()` pipeline (pipeline integration is later wiring work).
  *
  * Usage: `sbt "mengerApp/runMain menger.tools.SceneValidator <scene-file.scala>"`.
  */
object SceneValidator extends LazyLogging:

  enum Tag:
    case Ok, CompileErrors, LintFindings, Refused

  /** AD-5's tagged result contract names these variants literally in lowercase, hyphenated
    * form (`ok` | `compile-errors` | `lint-findings` | `refused`,
    * `ARCHITECTURE-SPINE.md`'s AD-5 rule) -- a consumer on the other side of this JSON (the
    * agent pipeline, `history.jsonl`) matches against those exact strings, not Scala's
    * PascalCase enum case names. `derives ReadWriter`'s default enum encoding would emit
    * `"Ok"`/`"CompileErrors"`/... instead, silently breaking that contract, so `Tag` gets an
    * explicit `ReadWriter` bridging the two representations. */
  object Tag:
    private def toWire(tag: Tag): String = tag match
      case Ok            => "ok"
      case CompileErrors => "compile-errors"
      case LintFindings  => "lint-findings"
      case Refused       => "refused"

    private def fromWire(wire: String): Tag = wire match
      case "ok"             => Ok
      case "compile-errors" => CompileErrors
      case "lint-findings"  => LintFindings
      case "refused"        => Refused
      case other            => sys.error(s"Unknown SceneValidator.Tag: '$other'")

    given ReadWriter[Tag] = readwriter[String].bimap[Tag](toWire, fromWire)

  /** One geometric invariant violation, carried structurally rather than flattened into
    * `messages` (review round 2): `PolytopeInvariants` documents `invariant` as "a short,
    * stable machine-readable tag", and a consumer that has to string-split `"tag: message"`
    * back apart is re-parsing something that was already structured. Mirrors
    * `menger.objects.higher_d.InvariantFinding` rather than serializing it directly, so the
    * geometry package stays free of a upickle dependency. */
  case class Finding(invariant: String, message: String) derives ReadWriter

  /** Bumped when this result's shape changes incompatibly, matching `CorpusManifest` and
    * `DslManifest`, which both carry one (review round 2). */
  val SchemaVersion = "1.0.0"

  /** `messages` stays first-class and unchanged -- it is what the existing consumers read.
    * `findings`, `scene` and `schemaVersion` are additive and defaulted, so every positional
    * `ValidationResult(tag, messages)` call site keeps working. */
  case class ValidationResult(
    tag: Tag,
    messages: List[String],
    findings: List[Finding] = Nil,
    scene: Option[String] = None,
    schemaVersion: String = SchemaVersion
  ) derives ReadWriter

  def main(args: Array[String]): Unit =
    val result = run(args) match
      case Right(r)      => r
      // A usage error is exactly what `refused` exists for: the pipeline was handed something
      // it cannot validate. Emitting the message through the logger and no JSON at all
      // (review round 2) broke run-sandboxed.sh's documented "prints a tagged JSON result"
      // contract for the one case most likely to be hit by a mis-wired caller.
      case Left(message) => ValidationResult(Tag.Refused, List(message))
    SceneValidatorMain.printResult(write(result, indent = 2))
    SceneValidatorMain.exitWith(exitCodeFor(result.tag))

  /** AD-5's four tags collapse onto three exit codes so a caller can branch without parsing
    * JSON (review round 2 -- previously every non-`ok` tag exited 1, making "the scene is
    * wrong, retry" indistinguishable from "the pipeline is broken, stop"). */
  private[tools] def exitCodeFor(tag: Tag): Int = tag match
    case Tag.Ok      => 0
    case Tag.Refused => 2
    case _           => 1

  /** Pure-ish entry point (the only I/O is loading/compiling the scene file) -- kept separate
    * from `main` so tests can observe results without triggering `sys.exit`, mirroring
    * `ManifestGenerator.run`/`CorpusExporter.run`. */
  def run(args: Array[String]): Either[String, ValidationResult] =
    args.headOption match
      case None       => Left("Usage: SceneValidator <scene-file.scala>")
      case Some(path) => Right(validate(new File(path)))

  /** Runs the full stage-1 + stage-4 gauntlet against `file` and returns AD-5's tagged
    * result. Does not throw for any failure a scene file can provoke: unexpected exceptions,
    * and the `Error`s a hostile or careless scene actually reaches (`StackOverflowError`,
    * `OutOfMemoryError`, `LinkageError` -- see the handlers below), are caught and reported
    * as a tagged result, never surfaced as a crash -- a scene file's job is to reveal defects
    * in the *scene*, not to take the validator down with it. A genuinely fatal JVM error
    * outside that set is still allowed to propagate.
    *
    * `preflightCheck` defaults to `RestrictedClasspath.preflight` but is an injectable seam
    * (review round 1 fix): the production classpath is always structurally complete in this
    * repo's own dev/test/container environments, so nothing exercised this method's
    * `Refused`-via-bad-classpath branch before -- a test now injects a forced failure here
    * rather than needing to actually corrupt the JVM's live classpath to reach it. */
  def validate(file: File, preflightCheck: () => Option[String] = RestrictedClasspath.preflight): ValidationResult =
    preflightCheck() match
      case Some(reason) =>
        ValidationResult(Tag.Refused, List(s"restricted classpath is not usable: $reason"))
      case None if !file.isFile =>
        // Checked directly rather than relying on SceneLoader's own message text: a scene
        // name that doesn't look like an existing file path (e.g. it was never written, or a
        // typo, or a directory rather than a file) falls through to SceneLoader's
        // *registry/reflection* lookup instead of its file-not-found branch
        // (`SceneLoader.isFilePath` requires the file to already exist to be treated as a
        // path at all), which would otherwise get misclassified as a scene defect instead of
        // AD-5's `refused` (a resource the pipeline was told to validate that simply isn't a
        // usable file). `isFile`, not `exists`, per review round 1: `exists()` is also true
        // for a directory, which `SceneLoader.load` has no defined behavior for.
        ValidationResult(Tag.Refused, List(s"scene file not found: ${file.getAbsolutePath}"))
      case None =>
        val scenePath = file.getAbsolutePath
        try
          (SceneLoader.load(scenePath) match
            case Left(err)     => classifyLoadFailure(err)
            case Right(loaded) => checkInvariants(loaded)
          ).copy(scene = Some(scenePath))
        catch
          // A `require()` rejection inside a compiled scene's static initializer (e.g.
          // `Sphere(size = -1f)`) surfaces as `ExceptionInInitializerError` -- a
          // `LinkageError`, which both `scala.util.Try` (used internally by
          // `SceneLoader.loadByReflectionWithLoader`) and `scala.util.control.NonFatal`
          // treat as fatal and let propagate rather than catch. It is not fatal to *this*
          // validator: a scene's own bad `require()` is exactly the kind of scene-content
          // defect this validator exists to report, not a reason to crash the process. Its
          // root cause is almost always the `require()`'s own `IllegalArgumentException`.
          case e: ExceptionInInitializerError =>
            val cause = Option(e.getCause).map(_.getMessage).getOrElse(e.getMessage)
            ValidationResult(
              Tag.LintFindings, List(s"scene construction failed: $cause"), scene = Some(scenePath)
            )
          // Review round 2: the scaladoc above promised "never throws", but only
          // `ExceptionInInitializerError` and `NonFatal` were handled. The three `Error`s
          // below are exactly the ones a hostile or merely careless scene provokes, and each
          // escaped as a bare stack trace with no tag on stdout.
          //   - StackOverflowError: infinite recursion in a scene's own `lazy val` -- the case
          //     run-sandboxed.sh's own comment anticipates. A scene defect: `lint-findings`.
          //   - OutOfMemoryError: a huge allocation under the container's `--memory` ceiling.
          //     Not decidably the scene's fault, and the JVM is now unreliable: `refused`.
          //   - LinkageError (incl. NoClassDefFoundError from a previously-failed
          //     initializer): an environment/classpath problem, never scene content:
          //     `refused`.
          case e: StackOverflowError =>
            ValidationResult(
              Tag.LintFindings,
              List(s"scene construction overflowed the stack (infinite recursion?): $e"),
              scene = Some(scenePath)
            )
          case e: OutOfMemoryError =>
            ValidationResult(
              Tag.Refused, List(s"validator ran out of memory: $e"), scene = Some(scenePath)
            )
          case e: LinkageError =>
            ValidationResult(
              Tag.Refused, List(s"validator classpath is broken: $e"), scene = Some(scenePath)
            )
          case NonFatal(e) =>
            logger.error(s"Unexpected failure validating $scenePath", e)
            ValidationResult(
              Tag.Refused,
              List(s"unexpected validator failure: ${e.getMessage}"),
              scene = Some(scenePath)
            )

  /** `SceneLoader.load`'s `Left` messages come from three distinct sources, each mapped to a
    * different AD-5 tag:
    *   - `SceneCompiler`'s own message (always starts with "Compilation of") -- a genuine
    *     Scala/DSL syntax error -- `CompileErrors`.
    *   - "Scene file not found: ..." -- the resource the pipeline was told to validate simply
    *     isn't there -- a pipeline-level `Refused`, not a defect in scene content that was
    *     never even read.
    *   - Everything else (no top-level object, no `scene`/`scene(Float)` member, or a
    *     `require()` precondition rejecting a compiled-and-loaded scene's construction
    *     parameters, caught by `SceneLoader`'s own `Try`) -- a scene-content defect the
    *     loader itself caught, not a compiler error -- `LintFindings`, matching the I/O
    *     matrix's "compile-errors or lint-findings" allowance for `require()` rejections. */
  private def classifyLoadFailure(err: String): ValidationResult =
    if err.startsWith("Compilation of") then
      ValidationResult(Tag.CompileErrors, List(err))
    else if err.startsWith("Scene file not found") then
      ValidationResult(Tag.Refused, List(err))
    else
      ValidationResult(Tag.LintFindings, List(err))

  private def checkInvariants(loaded: LoadedScene): ValidationResult =
    val scene = loaded match
      case LoadedScene.Static(s)   => s
      case LoadedScene.Animated(f) => f(0f)
    val findings = geometricFindings(scene)
    if findings.isEmpty then ValidationResult(Tag.Ok, Nil)
    else ValidationResult(
      Tag.LintFindings,
      findings.map(f => s"${f.invariant}: ${f.message}"),
      findings = findings.map(f => Finding(f.invariant, f.message))
    )

  /** Every `SceneObject` in `scene` (flat `objects` list, or the scene-graph `root`'s leaf
    * geometry when that's what the scene uses -- `Scene` requires at least one of the two)
    * that maps to a 4D `Mesh4D` via `MeshFactory.mesh4D` gets checked; anything else (3D
    * primitives, or free-form/lambda objects out of this story's scope per the `Never`
    * clause) is silently skipped -- "when applicable", per the story's own Code Map wording.
    */
  private def geometricFindings(scene: Scene): List[InvariantFinding] =
    val objects = scene.objects ++ scene.root.toList.flatMap(_.allLeafGeometry)
    objects
      .flatMap(obj => MeshFactory.mesh4D(obj.toObjectSpec))
      .flatMap(mesh => PolytopeInvariants.check(mesh))

/** Isolates the three process-level side effects `ArchitectureSpec` restricts to classes
  * whose name matches `.*Main.*` (writing to stdout/stderr, calling `sys.exit`) -- same
  * convention as `ManifestGeneratorMain`/`CorpusExporterMain`. */
private object SceneValidatorMain:
  // `Predef.print`, not `println`: this is the tool's real output contract (the JSON a
  // caller like run-sandboxed.sh parses from stdout), not a debug statement -- routing it
  // through `LazyLogging`'s logger instead would risk Logback's own formatting (timestamps,
  // level prefixes) polluting the JSON payload downstream tooling expects verbatim on
  // stdout, exactly the same reasoning `write(result, indent = 2)` already applies to keep
  // the payload machine-parseable.
  def printResult(json: String): Unit =
    print(json)
    print(System.lineSeparator())

  def exitWith(code: Int): Unit =
    if code != 0 then sys.exit(code)
