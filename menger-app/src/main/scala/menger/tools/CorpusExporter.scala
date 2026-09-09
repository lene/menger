package menger.tools

import java.io.IOException
import java.nio.file.Files
import java.nio.file.InvalidPathException
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardCopyOption

import scala.jdk.CollectionConverters.IteratorHasAsScala
import scala.util.Try
import scala.util.control.NonFatal

import com.typesafe.scalalogging.LazyLogging
import upickle.default.ReadWriter
import upickle.default.write

/** Generates a versioned JSON bundle of the renderer's DSL example-scene corpus -- the
  * few-shot corpus a scene-generation agent composes from alongside `ManifestGenerator`'s
  * capability manifest. One entry per source file directly under `examples/dsl/`.
  *
  * `SceneIndex.scala` is excluded: it is a registration index that aggregates other scenes'
  * `.scene` values into lookup lists, not a scene of its own. Nothing under `examples/dsl/common/`
  * is a standalone scene either (the scan is non-recursive) -- `Lighting.scala`/`Materials.scala`
  * there are shared presets already factored out, per dsl-surface.md's Corpus shape note.
  *
  * Usage: `sbt "mengerApp/runMain menger.tools.CorpusExporter [output-path] [source-dir]"`.
  * Defaults to `target/dsl-corpus.json` and `menger-app/src/main/scala/examples/dsl` (resolved
  * relative to the sbt working directory, i.e. the repo root) when not given.
  */
object CorpusExporter extends LazyLogging:

  private val SchemaVersion = "1.0.0"

  // Toolchain version pins (Always rule: no sbt-buildinfo -- a hardcoded constant is enough).
  // Kept in sync with ManifestGenerator's own pins -- see that file for the sync-with-build.sbt
  // note; duplicated here rather than shared because both are private to their own object.
  private val ScalaVersionPin = "3.8.3"
  private val OptixJniVersionPin = "0.3.3"
  private val MinDriverVersion = "580.65"

  private val DefaultOutputPath = "target/dsl-corpus.json"
  private val DefaultSourceDir = "menger-app/src/main/scala/examples/dsl"

  /** Not a standalone scene: `SceneIndex` is the registry, not an example. Shared presets live
    * under `examples/dsl/common/`, which is excluded structurally by the scan being
    * non-recursive -- move one up into `examples/dsl/` and it silently becomes a corpus
    * "scene", which `CorpusExporterSuite` now pins against (review round 2). */
  private val ExcludedFileNames = Set("SceneIndex.scala")

  case class SceneSource(name: String, path: String, source: String) derives ReadWriter

  case class CorpusManifest(
    schemaVersion: String,
    scalaVersion: String,
    optixJniVersion: String,
    minDriverVersion: String,
    scenes: List[SceneSource]
  ) derives ReadWriter

  def main(args: Array[String]): Unit =
    run(args) match
      case Right(()) => ()
      case Left(message) => CorpusExporterMain.reportFailureAndExit(message)

  /** Pure-ish entry point (the only I/O is reading the source files and writing the corpus) --
    * kept separate from `main` so tests can observe failures without triggering `sys.exit`,
    * mirroring `ManifestGenerator.run`. */
  def run(args: Array[String]): Either[String, Unit] =
    // Review round 2: both positionals were taken unvalidated, so `CorpusExporter --help`
    // wrote a file literally named `--help` and a third stray argument was silently ignored.
    val usage = "usage: CorpusExporter [output-path] [source-dir]"
    if args.length > 2 then Left(s"Error: too many arguments -- $usage")
    else if args.exists(_.startsWith("-")) then
      Left(s"Error: unexpected option '${args.find(_.startsWith("-")).getOrElse("")}' -- $usage")
    else
      runValidated(resolveOutputPath(args), resolveSourceDir(args))

  private def runValidated(outputPath: String, sourceDir: String): Either[String, Unit] =
    try
      buildCorpus(sourceDir).flatMap(writeCorpus(_, outputPath))
    catch
      case NonFatal(e) =>
        logger.error("Failed to build the DSL example-scene corpus", e)
        Left(s"Error: failed to build the DSL example-scene corpus: ${e.getMessage}")

  def resolveOutputPath(args: Array[String]): String =
    args.headOption.getOrElse(DefaultOutputPath)

  def resolveSourceDir(args: Array[String]): String =
    args.drop(1).headOption.getOrElse(DefaultSourceDir)

  private def buildCorpus(sourceDir: String): Either[String, CorpusManifest] =
    val dir = Paths.get(sourceDir)
    if !Files.isDirectory(dir) then
      Left(s"Error: source directory '$sourceDir' does not exist or is not a directory")
    else
      val stream = Files.list(dir)
      try
        val (unreadable, scenes) = stream.iterator().asScala
          .filter(p => Files.isRegularFile(p) && p.getFileName.toString.endsWith(".scala"))
          .filterNot(p => ExcludedFileNames.contains(p.getFileName.toString))
          .toList
          .sortBy(_.getFileName.toString)
          .map { p =>
            val fileName = p.getFileName.toString
            // A scene file that isn't valid UTF-8 previously failed the whole export with a
            // MalformedInputException whose message never named the offending file (review
            // round 2).
            Try(Files.readString(p)).toEither
              .left.map(e => s"unreadable scene file '$fileName': ${e.getMessage}")
              .map(src => SceneSource(
                name = fileName.stripSuffix(".scala"),
                // Relative to `sourceDir`, so the field carries locating information rather
                // than repeating `name + ".scala"` (review round 2).
                path = dir.relativize(p).toString,
                source = src
              ))
          }
          .partitionMap(identity)

        if unreadable.nonEmpty then
          Left(s"Error: ${unreadable.mkString("; ")}")
        else if scenes.isEmpty then
          Left(s"Error: no .scala scene files found in '$sourceDir' -- refusing to write an empty corpus")
        else
          Right(CorpusManifest(
            schemaVersion = SchemaVersion,
            scalaVersion = ScalaVersionPin,
            optixJniVersion = OptixJniVersionPin,
            minDriverVersion = MinDriverVersion,
            scenes = scenes
          ))
      finally
        stream.close()

  private def writeCorpus(corpus: CorpusManifest, outputPath: String): Either[String, Unit] =
    try
      val path: Path = Paths.get(outputPath)
      Option(path.getParent).foreach(Files.createDirectories(_))
      // AD-14, same as ManifestGenerator.writeManifest: this artifact crosses into the agent
      // domain read-only, so it is renamed into place rather than written in situ (review
      // round 2).
      val tmp = Files.createTempFile(
        Option(path.getParent).getOrElse(Paths.get(".")), ".dsl-corpus-", ".json.tmp"
      )
      Files.writeString(tmp, write(corpus, indent = 2))
      Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
      Right(())
    catch
      case e: IOException =>
        logger.error(s"Failed to write corpus to $outputPath", e)
        Left(s"Error: failed to write corpus to '$outputPath': ${e.getMessage}")
      case e: InvalidPathException =>
        logger.error(s"Invalid corpus output path: $outputPath", e)
        Left(s"Error: invalid output path '$outputPath': ${e.getMessage}")

/** Isolates the two process-level side effects `ArchitectureSpec` restricts to classes whose
  * name matches `.*Main.*` (writing to stderr, calling `sys.exit`) -- same convention as
  * `ManifestGeneratorMain`. */
private object CorpusExporterMain extends LazyLogging:
  def reportFailureAndExit(message: String): Unit =
    logger.error(message)
    sys.exit(1)
