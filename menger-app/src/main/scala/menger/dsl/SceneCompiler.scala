package menger.dsl

import java.io.File
import java.net.URLClassLoader
import java.nio.file.Files

import scala.util.Try

import com.typesafe.scalalogging.LazyLogging
import dotty.tools.dotc.Driver

object SceneCompiler extends LazyLogging:

  /** Compile a .scala file and return a ClassLoader over the output directory.
   *  Compiler errors are printed to stderr by the Dotty reporter; the Left message
   *  is a brief summary pointing the user there.
   */
  def compile(sourceFile: File): Either[String, ClassLoader] =
    val outputDir = Files.createTempDirectory("menger-scene-").toFile
    val cp        = currentClasspath
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

  @SuppressWarnings(Array("org.wartremover.warts.IsInstanceOf"))
  private def currentClasspath: String =
    def urlsFrom(cl: ClassLoader): Seq[java.net.URL] = cl match
      case ucl: URLClassLoader => ucl.getURLs.toSeq ++ urlsFrom(ucl.getParent)
      // scalafix:off DisableSyntax.null
      case null                => Seq.empty
      // scalafix:on DisableSyntax.null
      case other               => urlsFrom(other.getParent)

    val loaderUrls = urlsFrom(Thread.currentThread.getContextClassLoader)

    val sysPropEntries = System.getProperty("java.class.path", "")
      .split(File.pathSeparator)
      .filter(_.nonEmpty)
      .toSeq

    val loaderPaths = loaderUrls.flatMap(u => Try(new File(u.toURI).getAbsolutePath).toOption)

    (loaderPaths ++ sysPropEntries).distinct.mkString(File.pathSeparator)
