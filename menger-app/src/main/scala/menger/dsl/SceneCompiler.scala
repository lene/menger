package menger.dsl

import java.io.File
import java.net.URLClassLoader
import java.nio.file.Files

import com.typesafe.scalalogging.LazyLogging
import dotty.tools.dotc.Driver

object SceneCompiler extends LazyLogging:

  /** Compile a .scala file and return a ClassLoader over the output directory.
   *  Compiler errors are printed to stderr by the Dotty reporter; the Left message
   *  is a brief summary pointing the user there.
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
      Left(s"Compilation of '${sourceFile.getName}' failed (see compiler output above)")
    else
      Right(URLClassLoader(
        Array(outputDir.toURI.toURL),
        Thread.currentThread.getContextClassLoader
      ))
