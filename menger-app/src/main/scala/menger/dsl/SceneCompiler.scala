package menger.dsl

import java.io.File
import java.net.URLClassLoader
import java.nio.file.Files

import com.typesafe.scalalogging.LazyLogging
import dotty.tools.dotc.Driver
import dotty.tools.dotc.reporting.Diagnostic.Error

object SceneCompiler extends LazyLogging:

  /** Compile a .scala file and return a ClassLoader over the output directory.
   *
   *  Compiles against `RestrictedClasspath.build()` -- the DSL-surface-scoped classpath
   *  (AD-4 rule 2) -- rather than the full unrestricted classpath every jar on the current
   *  JVM would otherwise expose to a compiled scene file. See `RestrictedClasspath` for what
   *  is included/excluded and why.
   */
  def compile(sourceFile: File): Either[String, ClassLoader] =
    val outputDir = Files.createTempDirectory("menger-scene-").toFile
    val cp        = RestrictedClasspath.build()
    logger.debug(s"Compiling ${sourceFile.getAbsolutePath} → ${outputDir.getAbsolutePath}")

    val args = Array(
      "-classpath", cp,
      "-d",         outputDir.getAbsolutePath,
      sourceFile.getAbsolutePath
    )

    val reporter = new Driver().process(args)

    if reporter.hasErrors then
      Left(formatErrors(sourceFile, reporter.allErrors))
    else
      Right(URLClassLoader(
        Array(outputDir.toURI.toURL),
        Thread.currentThread.getContextClassLoader
      ))

  /** Renders the Dotty reporter's own buffered diagnostics (`Reporter.allErrors` -- populated
    * by the base `Reporter` class regardless of which concrete reporter ran the compile, so
    * this needs no custom `StoreReporter`/`Context` plumbing) into the `Left` a caller actually
    * gets back. Previously this was a generic placeholder pointing at "compiler output above" --
    * output that a sandboxed subprocess caller (`SceneValidator`, AD-18) never sees, since the
    * reporter's own console output is on the compiler JVM's stderr, not folded into the JSON
    * result. `Diagnostic.message` needs no `Context` argument (already ANSI-stripped by its own
    * `override def message`), so this stays a plain, directly testable string transform. */
  private def formatErrors(sourceFile: File, errors: List[Error]): String =
    val details = errors.map { err =>
      if err.pos.exists then s"line ${err.pos.line + 1}: ${err.message}"
      else err.message
    }
    s"Compilation of '${sourceFile.getName}' failed:\n${details.mkString("\n")}"
