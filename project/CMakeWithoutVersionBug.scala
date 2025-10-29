package com.github.sbt.jni.build

import sbt._
import sys.process._

/**
 * Custom CMake build tool that fixes the sbt-jni version parsing bug.
 *
 * sbt-jni v1.6.0-1.7.1 has a bug where it passes the cmake version number
 * (e.g., "328" for CMake 3.28.3) as an extra argument to the cmake command,
 * causing a spurious warning:
 *
 *   CMake Warning:
 *     Ignoring extra path from command line:
 *      "/path/to/build/328"
 *
 * This class is identical to the original CMake class except it removes
 * the spurious `cmakeVersion.toString` argument from the configure() method.
 *
 * Unfortunately, we can't inherit and override just configure() because
 * CMake.getInstance() returns an anonymous Instance class that can't be extended.
 *
 * Bug location in sbt-jni:
 *   plugin/src/main/scala/com/github/sbt/jni/build/CMake.scala:44-47
 *
 * See: https://github.com/sbt/sbt-jni/blob/v1.7.1/plugin/src/main/scala/com/github/sbt/jni/build/CMake.scala#L44-L47
 */
class CMakeWithoutVersionBug(protected val configuration: Seq[String])
    extends BuildTool with ConfigureMakeInstall {

  override val name = "CMake"

  override def detect(baseDirectory: File) = baseDirectory.list().contains("CMakeLists.txt")

  override protected def templateMappings = Seq(
    "/com/github/sbt/jni/templates/CMakeLists.txt" -> "CMakeLists.txt"
  )

  override def getInstance(baseDir: File, buildDir: File, logger: Logger) = new Instance {

    override def log = logger
    override def baseDirectory = baseDir
    override def buildDirectory = buildDir

    def cmakeProcess(args: String*): ProcessBuilder = Process("cmake" +: args, buildDirectory)

    lazy val cmakeVersion =
      cmakeProcess("--version").lineStream.head
        .split("\\s+")
        .last
        .split("\\.") match {
        case Array(maj, min, rev) =>
          logger.info(s"Using CMake version $maj.$min.$rev")
          maj.toInt * 100 + min.toInt
        case _ => -1
      }

    def parallelOptions: Seq[String] =
      if (cmakeVersion >= 312) Seq("--parallel", parallelJobs.toString)
      else Seq.empty

    override def configure(target: File) = {
      // FIXED: Original sbt-jni code incorrectly passes cmakeVersion.toString here
      // This caused cmake to see "328" as an extra path argument, generating a warning
      cmakeProcess(
        (s"-DCMAKE_INSTALL_PREFIX:PATH=${target.getAbsolutePath}" +: configuration) ++ Seq(
          baseDirectory.getAbsolutePath  // Only pass source directory, not version
        ): _*
      )
    }

    override def clean(): Unit = cmakeProcess(
      "--build",
      buildDirectory.getAbsolutePath,
      "--target",
      "clean"
    ).run(log)

    override def make(): ProcessBuilder = cmakeProcess(
      Seq("--build", buildDirectory.getAbsolutePath) ++ parallelOptions: _*
    )

    override def install(): ProcessBuilder =
      if (cmakeVersion >= 315) cmakeProcess("--install", buildDirectory.getAbsolutePath)
      else Process("make install", buildDirectory)
  }
}

object CMakeWithoutVersionBug {
  val DEFAULT_CONFIGURATION = Seq("-DCMAKE_BUILD_TYPE=Release", "-DSBT:BOOLEAN=true")

  def make(configuration: Seq[String] = DEFAULT_CONFIGURATION): BuildTool =
    new CMakeWithoutVersionBug(configuration)

  lazy val release: BuildTool = make()
}
